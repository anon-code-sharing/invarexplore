import os
import torch
import pickle

from datasets import load_dataset
from copy import deepcopy
from awq.quantize.pre_quant import run_awq, apply_awq, get_blocks
from accelerate.utils.modeling import get_balanced_memory
from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    dispatch_model,
    load_checkpoint_in_model,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from awq.quantize.quantizer import (
    pseudo_quantize_model_weight,
    real_quantize_model_weight,
)


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


def create_rotation_matrix(angles):

    d = len(angles) // 2

    # Prepare cosines and sines of the angles
    cos_theta = torch.cos(angles)
    sin_theta = torch.sin(angles)

    # Create 2x2 rotation blocks
    rotation_blocks = torch.stack([
        torch.stack([cos_theta, -sin_theta], dim=1),
        torch.stack([sin_theta, cos_theta], dim=1)
    ], dim=1)  # Shape: (num_blocks, 2, 2)

    rotation_matrix = torch.block_diag(*rotation_blocks)

    return rotation_matrix


@torch.no_grad
def build_model_and_enc(
    model_path=None,
    return_full=False,
    w_bit=2,
    q_group_size=None,
    zero_point=True,
    load_awq='',
    load_invars='',
    backend='fake',
):

    # get quantization config (apart from w_bit)
    q_config = {
        "zero_point": zero_point,  # by default True
        "q_group_size": q_group_size,  # whether to use group quantization
    }
    print("Quantization config:", q_config)

    if not os.path.exists(model_path):  # look into ssd
        raise FileNotFoundError(f"{model_path} not found!")
    print(f"* Building model {model_path}")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    config.use_cache = False
    if "mpt" in config.__class__.__name__.lower():
        enc = AutoTokenizer.from_pretrained(
            config.tokenizer_name, trust_remote_code=True
        )
    else:
        enc = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, trust_remote_code=True
        )

    # Init model on CPU:
    kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
    vila_10_quant_mode = ("llava" in model_path.lower() or "vila" in model_path.lower()) and not vila_15
    if not vila_10_quant_mode:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, trust_remote_code=True, **kwargs
        )

    model.eval()

    if load_awq:
        print("Loading pre-computed AWQ results from", load_awq)
        awq_results = torch.load(load_awq, map_location="cpu")
        apply_awq(model, awq_results)

    if load_invars:
        print('Loading invars from', load_invars)

        with open(load_invars, mode='rb') as f:
            invars = pickle.load(f)

        for layer_i in range(model.config.num_hidden_layers):

            fc1 = model.model.decoder.layers[layer_i].fc1.cuda()
            fc2 = model.model.decoder.layers[layer_i].fc2.cuda()

            # ----- apply invariance-----
            if 'rotations' in invars.keys():
                rotation = invars['rotations'][layer_i].cuda()
                R = create_rotation_matrix(rotation)
                fc1.weight.data = R.T @ fc1.weight
                fc1.bias.data = R.T @ fc1.bias
                fc2.weight.data = fc2.weight @ R
            if 'scales' in invars.keys():
                scale = invars['scales'][layer_i].cuda()
                fc1.weight.div_(scale.view(-1, 1))
                fc1.bias.div_(scale.view(-1))
                fc2.weight.mul_(scale.view(1, -1))
            if 'orders' in invars.keys():
                order = invars['orders'][layer_i]
                fc1.weight.data = fc1.weight.data[order, :]
                fc1.bias.data = fc1.bias.data[order]
                fc2.weight.data = fc2.weight.data[:, order]
            # ---------------------------

            fc1.cpu()
            fc2.cpu()

    # weight quantization
    if return_full:
        full_model = deepcopy(model)

    if w_bit < 16:
        if backend == 'fake':
            pseudo_quantize_model_weight(model, w_bit=w_bit, q_config=q_config)
        elif backend =='real':
            real_quantize_model_weight(model, w_bit=w_bit, q_config=q_config)

    # Move the model to GPU (as much as possible) for LM evaluation
    kwargs = {
        "max_memory": get_balanced_memory(
            model, None
        )
    }

    if return_full:
        return full_model, model, enc
    return model, enc


def get_calib_batches(data="pileval", tokenizer=None, n_batches=1, block_size=512):
    if data == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    else:
        raise NotImplementedError
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    n_tokens = 0
    required_tokens = n_batches * block_size
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        n_tokens += len(line_encoded)
        if n_tokens >= required_tokens:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks, {n_run} samples used")
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]
