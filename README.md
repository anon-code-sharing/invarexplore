# Exploring Model Invariance for Ultra-Low-Bit Quantization

This code base is largely based on the [AWQ repo](https://github.com/mit-han-lab/llm-awq).

## Setup
First, install the necessary packages with these
```
pip install -e .
pip install -r requirements.txt
```

## Run InvarExplore

The main search algorithm is located at awq/search.py, which you can run with
```
python -m awq.search --model_path PATH/TO/MODEL \
    --load_awq PATH/TO/AWQ/PARAMETERS \
    --tasks wikitext \
    --w_bit 2 --q_group_size 128 \
    --q_backend fake \
    --log_name NAME \
    --alpha 0.1 \
    --num_batches 32 \
    --swap_frac 0.1 \
    --seed 0 \
    --invariances rsp \
    --scale_std 1e-2 \
    --rotation_std 1e-5 \
    --num_layers 10 \
```
Note that --invariances specifies the types of invariance to optimize for. E.g., "rsp" means search for all, and "r" means search for rotation only.

## Evaluation

For language modeling, run
```
python -m awq.evaluate --model_path PATH/TO/MODEL \
    --w_bit 2 --q_group_size 128
    --task=wikitext2/c4
```

For reasoning tasks, we use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). 
