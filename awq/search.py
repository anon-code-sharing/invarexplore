from lm_eval import evaluator, tasks
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import argparse
import os
import json
import math
from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    dispatch_model,
    load_checkpoint_in_model,
)
from accelerate.utils.modeling import get_balanced_memory
from awq.utils.parallel import auto_parallel
from awq.quantize.pre_quant import run_awq, apply_awq, get_blocks
from awq.quantize.quantizer import (
    pseudo_quantize_model_weight,
    real_quantize_model_weight,
)
from awq.utils.lm_eval_adaptor import LMEvalAdaptor
from awq.utils.utils import simple_dispatch_model
from datasets import load_dataset
from torch import nn
import tqdm

import random
from awq.quantize.quantizer import pseudo_quantize_tensor
import gc
import wandb
from dateutil import tz
from pathlib import PosixPath
import datetime
import numpy as np
import sys
from functools import partial
import pickle
from awq.utils.simple_utils import build_model_and_enc, get_calib_batches, create_rotation_matrix
from awq.evaluate import evaluate
import torch.nn.functional as F
from copy import deepcopy

class Logger():
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')
        self.encoding = 'UTF-8'

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="path of the hf model")
parser.add_argument("--tasks", default=None, type=str)
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument("--num_fewshot", type=int, default=0)
parser.add_argument('--log_name', type=str, default='')
parser.add_argument('--swap_frac', type=float, default=1e-2)
parser.add_argument('--search_unit', type=str, choices=['model', 'layer'], default='model')
parser.add_argument('--log_root_dir', type=str, default='')
parser.add_argument('--no_save', action='store_true')
parser.add_argument('--save_type', type=str, choices=['invars', 'model'], default='invars')
parser.add_argument('--wandb_project', type=str, default='search')
parser.add_argument('--alpha', type=float, default=1e-1, help='the strength of layer loss')
parser.add_argument('--num_batches', type=int, default=1)
parser.add_argument('--load_invars', type=str, default='', help='path of the invariances')
parser.add_argument('--num_layers', type=int, default=-1, help='-1 means learn from all layers')
parser.add_argument('--invariances', type=str, default='rsp', help='r: rotation, s: scaling, p: permuation')
parser.add_argument('--scale_std', type=float, default=1e-2)
parser.add_argument('--rotation_std', type=float, default=1e-6)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--teacher', type=str, default='')

# model config
parser.add_argument("--parallel", action="store_true", help="enable model parallelism")
# max memory to offload larger models to CPU
parser.add_argument(
    "--max_memory",
    type=str,
    nargs="*",
    help="List of device_id:max_memory pairs to be parsed into a dictionary; "
    + "Example: 0:10GiB 1:10GiB cpu:30GiB; "
    + "mode details here: "
    + "https://huggingface.co/docs/accelerate/usage_guides/big_modeling",
)
parser.add_argument(
    "--auto_parallel",
    action="store_true",
    help="automatically set parallel and batch_size",
)
# quantization config
parser.add_argument("--w_bit", type=int, default=None)
parser.add_argument("--q_group_size", type=int, default=-1)
parser.add_argument("--no_zero_point", action="store_true", help="disable zero_point")
parser.add_argument("--q_backend", type=str, default="fake", choices=["fake", "real"])
# save/load real quantized weights
parser.add_argument("--dump_quant", type=str, default=None, help="save quantized model")
parser.add_argument("--dump_fake", type=str, default=None, help="save fake-quantized model")
parser.add_argument("--load_quant", type=str, default=None, help="load quantized model")
# apply/save/load awq
parser.add_argument("--run_awq", action="store_true", help="perform awq search process")
parser.add_argument(
    "--dump_awq", type=str, default=None, help="save the awq search results"
)
parser.add_argument(
    "--load_awq", type=str, default=None, help="load the awq search results"
)
parser.add_argument(
    "--vila-15",
    action="store_true",
    help="quantizing vila 1.5",
)
args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

if args.auto_parallel:
    gpu_list = auto_parallel(args)


def shuffle_subset(order, subset_size):
    # Select random indices for the subset
    indices = np.random.choice(len(order), size=subset_size, replace=False)

    # Extract the subset based on selected indices
    subset = [order[i] for i in indices]

    # Shuffle the subset
    subset = [subset[-1]] + subset[:-1]

    # Create a copy of the original order to modify
    modified_order = order[:]

    # Replace the subset positions with shuffled values
    for idx, new_value in zip(indices, subset):
        modified_order[idx] = new_value

    return modified_order

def shuffle_indices(order, indices):
    # Extract the subset based on selected indices
    subset = [order[i] for i in indices]

    # Shuffle the subset
    subset = [subset[-1]] + subset[:-1]

    # Create a copy of the original order to modify
    modified_order = order[:]

    # Replace the subset positions with shuffled values
    for idx, new_value in zip(indices, subset):
        modified_order[idx] = new_value

    return modified_order

@torch.no_grad()
def w_quantize_func(p):
    return pseudo_quantize_tensor(p, n_bit=args.w_bit,
        zero_point=not args.no_zero_point, q_group_size=args.q_group_size)


@torch.no_grad()
def compute_layer_loss(outputs_1, outputs_2):
    diff = torch.stack(outputs_1) - torch.stack(outputs_2)
    diff = torch.clamp(diff, min=-100, max=100)
    layer_loss = diff.pow(2).mean().item()
    return layer_loss


@torch.no_grad()
def compute_early_loss(model, calib_samples, outputs):
    early_losses = []

    for layer_i in range(len(outputs)):

        hidden_states = outputs[layer_i]
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)

        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)

        logits = model.lm_head(hidden_states)
        shift_logits = logits[:, :-1, :]
        shift_labels = calib_samples[:, 1:]
        recalc_ce_loss = float(CE_LOSS_FUNCTION(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1)))
        early_losses.append(recalc_ce_loss)

    return sum(early_losses) / len(early_losses)


CE_LOSS_FUNCTION = nn.CrossEntropyLoss()
@torch.no_grad()
def compute_ce_loss(model, calib_samples):
    logits = model(calib_samples).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = calib_samples[:, 1:]
    ce_loss = float(CE_LOSS_FUNCTION(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1)))
    return ce_loss


@torch.no_grad()
def dialated_subset(A, number):
    step = len(A) / number
    # select from the end
    return [A[len(A) - 1 - math.ceil(i * step)] for i in range(number)]
    # return [math.ceil(i * step)] for i in range(number)]


@torch.no_grad()
def main():

    tzone = tz.gettz('America/Edmonton')
    timestamp = datetime.datetime.now().astimezone(tzone).strftime('%Y-%m-%d_%H:%M:%S')

    if args.log_name:
        log_name = timestamp + '_' + args.log_name
    else:
        log_name = timestamp

    log_dir = PosixPath(args.log_root_dir, log_name)
    log_dir.mkdir()

    # logging
    log_txt_path = PosixPath(log_dir, timestamp + '.log')
    logger = Logger(log_txt_path)
    sys.stdout = logger
    sys.stderr = logger

    wandb.init(project=args.wandb_project, name=log_name)

    if args.output_path is not None and os.path.exists(args.output_path):
        # print(f"Results {args.output_path} already generated. Exit.")
        print(f"Results {args.output_path} already generated. Overwrite.")
        # exit()

    if args.dump_awq and os.path.exists(args.dump_awq):
        print(f"Found existing AWQ results {args.dump_awq}, exit.")
        exit()

    def search_model():

        # a hack here to auto set model group
        full_model, model, enc = build_model_and_enc(
            model_path=args.model_path,
            return_full=True,
            w_bit=args.w_bit,
            q_group_size=args.q_group_size,
            zero_point=not args.no_zero_point,
            load_awq=args.load_awq,
            load_invars=args.load_invars,
        )

        model.eval()
        full_model.eval()
        model.cpu()
        full_model.cpu()

        calib_samples = get_calib_batches(
            data='pileval', tokenizer=enc, n_batches=args.num_batches, block_size=512
        )
        calib_samples = torch.cat(calib_samples, dim=0).cuda()

        # caching inputs/outputs
        hook_enabled = True  # hook variable
        def cache_input_hook(m, x, args, y, cache):
            if hook_enabled:
                cache.append(y[0])

        full_outputs = []  # for FP16 model
        outputs = []  # for quantized model

        if args.teacher:
            real_full_model = full_model
            full_model = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.float16)
            full_model.eval()
            full_model.cpu()

        # set up caching for full model
        full_layers = get_blocks(full_model)
        if args.num_layers != -1:
            full_layers = dialated_subset(full_layers, args.num_layers)
        full_handles = []
        for full_layer in full_layers:
            full_handles.append(full_layer.register_forward_hook(
                partial(cache_input_hook, cache=full_outputs), with_kwargs=True))

        full_model.cuda()
        full_model(calib_samples)
        full_model.cpu()
        model.cuda()

        for h in full_handles:
            h.remove()

        if args.teacher:
            del full_model
            full_model = real_full_model

        # set up cahcing for quantized model
        layers = get_blocks(model)
        if args.num_layers != -1:
            layers = dialated_subset(layers, args.num_layers)
        handles = []
        for layer in layers:
            handles.append(layer.register_forward_hook(
                partial(cache_input_hook, cache=outputs), with_kwargs=True))

        num_layers = model.config.num_hidden_layers
        hdn_dim = model.config.hidden_size
        ffn_dim = model.config.ffn_dim
        num_accepts = 0
        swap_size = round(ffn_dim * args.swap_frac)
        assert swap_size > 1
        print('>>> swap size:', swap_size)

        orig_ce_loss = compute_ce_loss(model, calib_samples)
        orig_ee_loss = compute_early_loss(model, calib_samples, outputs)
        orig_layer_loss = compute_layer_loss(full_outputs, outputs)
        outputs.clear()  # delete quantized outputs
        best_loss = 1 + args.alpha

        if 'r' in args.invariances:
            rotations = [torch.zeros(ffn_dim // 2, device='cuda', dtype=torch.float16) for _ in range(num_layers)]
        if 's' in args.invariances:
            scales = [torch.ones(ffn_dim, device='cuda', dtype=torch.float16) for _ in range(num_layers)]
        if 'p' in args.invariances:
            orders = [list(range(ffn_dim)) for _ in range(num_layers)]

        # evaluate the model w/o saving partial inputs
        hook_enabled = False
        results = evaluate(model, enc)
        hook_enabled = True

        wandb.log({'test_ppl': results['ppl']}, step=0)
        print(f'step: {0}, test_pll: {results["ppl"]:.3f}')

        for step in range(1, 1000000000):

            # ----- sample a new order -----
            sampled_layer = random.randint(0, num_layers - 1)

            # sample a layer
            full_fc1 = full_model.model.decoder.layers[sampled_layer].fc1
            full_fc2 = full_model.model.decoder.layers[sampled_layer].fc2

            fc1 = model.model.decoder.layers[sampled_layer].fc1
            fc2 = model.model.decoder.layers[sampled_layer].fc2

            # save to revert back
            old_fc1_weight = fc1.weight.data.clone()
            old_fc1_bias = fc1.bias.data.clone()
            old_fc2_weight = fc2.weight.data.clone()

            # default actions
            if 'r' in args.invariances:
                rotation = deepcopy(rotations[sampled_layer])
            if 's' in args.invariances:
                scale = deepcopy(scales[sampled_layer])
            if 'p' in args.invariances:
                order = deepcopy(orders[sampled_layer])

            # ----- sample an action -----
            # action = random.choice(args.invariances)
            # if action == 'r':
            #     # sample a rotation
            #     rotation = torch.normal(mean=rotation, std=args.rotation_std)
            # elif action == 's':
            #     # sample a scale
            #     scale = torch.normal(mean=scale, std=args.scale_std)
            # elif action == 'p':
            #     # sample a subset to swap
            #     order = shuffle_subset(order, swap_size)

            # rotation happens in pairs of dimensions
            rotation_indices = np.random.choice(ffn_dim // 2, size=swap_size // 2, replace=False)
            # match the neurons that are rotated, i.e., rotation_indices and their adjacent neurons
            full_indices = np.concatenate((rotation_indices * 2, rotation_indices * 2 + 1))
            np.random.shuffle(full_indices)

            # Note here torch.normal modifies the existing rotation/scale, rather than generating a new one
            if 'r' in args.invariances:
                rotation[rotation_indices] = torch.normal(mean=rotation[rotation_indices], std=args.rotation_std)
            if 's' in args.invariances:
                scale[full_indices] = torch.normal(mean=scale[full_indices], std=args.scale_std)
            if 'p' in args.invariances:
                order = shuffle_indices(order, full_indices)

            # ----- apply invariance-----
            fc1.weight.data = full_fc1.weight.cuda()
            fc1.bias.data = full_fc1.bias.cuda()
            fc2.weight.data = full_fc2.weight.cuda()
            if 'r' in args.invariances:
                R = create_rotation_matrix(rotation)
                fc1.weight.data = R.T @ fc1.weight
                fc1.bias.data = R.T @ fc1.bias
                fc2.weight.data = fc2.weight @ R
            if 's' in args.invariances:
                fc1.weight.div_(scale.view(-1, 1))
                fc1.bias.div_(scale.view(-1))
                fc2.weight.mul_(scale.view(1, -1))
            if 'p' in args.invariances:
                fc1.weight.data = fc1.weight.data[order, :]
                fc1.bias.data = fc1.bias.data[order]
                fc2.weight.data = fc2.weight.data[:, order]
            fc1.weight.data = w_quantize_func(fc1.weight)
            fc2.weight.data = w_quantize_func(fc2.weight)
            # ---------------------------

            # compute the losses
            ce_loss = compute_ce_loss(model, calib_samples)
            # ee_loss = compute_early_loss(model, calib_samples, outputs)
            layer_loss = compute_layer_loss(full_outputs, outputs)
            loss = ce_loss/orig_ce_loss + args.alpha * layer_loss/orig_layer_loss
            outputs.clear()  # reset quantized outputs

            if loss < best_loss:
                best_loss = loss

                # if action == 'r':
                #     rotations[sampled_layer] = rotation
                # elif action == 's':
                #     scales[sampled_layer] = scale
                # elif action == 'p':
                #     orders[sampled_layer] = order

                if 'r' in args.invariances:
                    rotations[sampled_layer] = rotation
                if 's' in args.invariances:
                    scales[sampled_layer] = scale
                if 'p' in args.invariances:
                    orders[sampled_layer] = order

                num_accepts += 1

                # evaluate the model w/o saving partial inputs
                hook_enabled = False
                results = evaluate(model, enc)
                hook_enabled = True

                # saving
                if not args.no_save:
                    save_path = PosixPath(str(log_dir), 'best')
                    if not save_path.exists():
                        save_path.mkdir()
                    if args.save_type == 'model':
                        full_model.save_pretrained(save_path)
                        enc.save_pretrained(save_path)
                    elif args.save_type == 'invars':
                        invars = {}
                        if 'r' in args.invariances:
                            invars['rotations'] = rotations
                        if 's' in args.invariances:
                            invars['scales'] = scales
                        if 'p' in args.invariances:
                            invars['orders'] = orders
                        with open(PosixPath(save_path, 'ckpt.pkl'), mode='wb') as f:
                            pickle.dump(invars, f)

                    with open(PosixPath(save_path, 'METADATA'), mode='w') as f:
                        f.write(f'step: {step}, loss: {loss:.3f} test_pll: {results["ppl"]:.3f}')

                wandb.log({'test_ppl': results['ppl']}, step=step)
                print(f'step: {step}, test_pll: {results["ppl"]:.3f}')

            else:
                # revert to the original weights
                fc1.weight.data = old_fc1_weight
                fc1.bias.data = old_fc1_bias
                fc2.weight.data = old_fc2_weight

            accept_ratio = num_accepts / (step+1)
            wandb.log({'loss': loss, 'ce_loss': ce_loss, 'layer_loss': layer_loss,
                       'num_accepts': num_accepts, 'accept_ratio': accept_ratio}, step=step)
            print(f'step: {step}, loss: {loss:.3f}, accept fraction: {accept_ratio:.3f}')

            # torch.cuda.empty_cache()
            # gc.collect()
        # --------------------------------------

    search_model()

if __name__ == "__main__":
    main()
