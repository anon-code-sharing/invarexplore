import torch
import tqdm
import argparse
from torch import nn
from datasets import load_dataset
from awq.utils.simple_utils import build_model_and_enc
from pathlib import PosixPath


def evaluate(model, enc, task='wikitext2', rebuild=False):

    cache_file = PosixPath('DIR-TO-CACHE', f'{task}_testenc.pt')
    if not cache_file.exists() or rebuild:
        if task == 'wikitext2':
            testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            testenc = enc("\n\n".join(testenc["text"]), return_tensors="pt")
        elif task == 'c4':
            testenc = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
            testenc = enc("\n\n".join(testenc["text"]), return_tensors="pt")
        torch.save(testenc, cache_file)
    else:
        testenc = torch.load(cache_file)
    if task == 'c4':
        testenc['input_ids'] = testenc['input_ids'][:, :1000000]
        testenc['attention_mask'] = testenc['attention_mask'][:, :1000000]
    model.seqlen = 2048
    testenc = testenc.input_ids.to(model.device)
    nsamples = testenc.numel() // model.seqlen
    model = model.eval()
    nlls = []
    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
            model.device
        )
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[
            :, (i * model.seqlen) : ((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    results = {"ppl": ppl.item()}

    return results


if __name__ == '__main__':

    parser =argparse.ArgumentParser('')
    parser.add_argument('--model_path', type=str, required=True, default='')
    parser.add_argument('--w_bit', type=int, default=2)
    parser.add_argument("--q_group_size", type=int, default=128)
    parser.add_argument("--no_zero_point", action="store_true", help="disable zero_point")
    parser.add_argument('--load_awq', type=str, default='')
    parser.add_argument('--load_invars', type=str, default='')
    parser.add_argument('--task', choices=['wikitext2', 'c4'], default='wikitext2')
    parser.add_argument('--rebuild', action='store_true')

    args = parser.parse_args()

    model, enc = build_model_and_enc(
        model_path=args.model_path,
        return_full=False,
        w_bit=args.w_bit,
        q_group_size=args.q_group_size,
        zero_point=not args.no_zero_point,
        load_awq=args.load_awq,
        load_invars=args.load_invars,
    )

    model.cuda()
    results = evaluate(model, enc, task=args.task, rebuild=args.rebuild)

    print(results)
