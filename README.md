# Deconfounded Causality-aware Parameter-Efficient Fine-Tuning (DCA)

This repository contains the code for **Deconfounded Causality-aware Parameter-Efficient Fine-Tuning for Problem-Solving Improvement of LLMs** ([arXiv:2409.02686](https://arxiv.org/abs/2409.02686)).

## Overview

Large Language Models (LLMs) are strong at instruction following, but often struggle with *reasoning-intensive* tasks (e.g., math and physics). In the paper, we:
1. investigate whether the model’s behavior is truly reasoning (via attention/representation visualizations),
2. formulate the reasoning process as a causal framework to explain observed failure modes, and
3. propose **Deconfounded Causal Adaptation (DCA)**, a **parameter-efficient fine-tuning (PEFT)** method that improves problem-solving capability by encouraging the model to extract general problem-solving skills and apply them across questions.

In this codebase, DCA is implemented as an adapter-style PEFT method built on top of the open-source **LLaMA-Adapter** project: https://github.com/opengvlab/llama-adapter

## What’s included

* Inference with an adapter-augmented LLaMA backbone: `example.py` / `generate.sh`
* Fine-tuning scripts (adapter-only) for reasoning tasks: `alpaca_finetuning_v1/*.py`
* The LLaMA implementation with adapter injection: `llama/`

## Environment setup

Create an environment with PyTorch + CUDA (or CPU for small tests), then install dependencies:

```bash
# from repo root
pip install -r requirements.txt
pip install -e .
```

The fine-tuning scripts under `alpaca_finetuning_v1/` use the same adapter components and depend on PyTorch plus common utilities declared in `requirements.txt`.

## Download LLaMA checkpoints + tokenizer

Inference and fine-tuning expect a directory structure like:

```text
TARGET_FOLDER/
  7B/
    consolidated.00.pth
    params.json
    checklist.chk
  tokenizer.model
  tokenizer_checklist.chk
```

Edit `download.sh` and fill in:

* `PRESIGNED_URL` (provided by the project authors via their form/email),
* `TARGET_FOLDER` (where files should be downloaded),
* `MODEL_SIZE` (comma-separated list, e.g. `7B,13B,30B,65B`).

Then run:

```bash
bash download.sh
```

## Run inference

### Quick demo

`generate.sh` runs `example.py` with `torchrun`.

```bash
bash generate.sh
```

### General command

```bash
torchrun --nproc_per_node <MP> example.py \
  --ckpt_dir <TARGET_FOLDER>/<MODEL_SIZE> \
  --tokenizer_path <TARGET_FOLDER>/tokenizer.model \
  --adapter_path <PATH_TO_ADAPTER_CHECKPOINT>.pth
```

Notes:

* `<MP>` should match the number of model shards for that model size (see your LLaMA-Adapter setup instructions; e.g., 7B uses `--nproc_per_node 1`).
* `adapter_path` is typically the file produced by fine-tuning (see next section).
* `example.py` uses `fire` and will switch to interactive input if no prompts are provided.

## Fine-tune with DCA (adapter-only)

Fine-tuning scripts live in `alpaca_finetuning_v1/` and train only the adapter parameters (and attention gating parameters, depending on the adapter variant).

### Adapter checkpoints produced by training

Training writes adapter checkpoints to `--output_dir` as:

* `checkpoint-<epoch>.pth`

These files can be passed directly as `--adapter_path` to `example.py`.

### Example: math word problems

One typical training entrypoint is `alpaca_finetuning_v1/finetuning_math10k_case2.py` (run from the `alpaca_finetuning_v1/` directory because scripts use relative paths like `data/template_train.json`):

```bash
cd alpaca_finetuning_v1

torchrun --nproc_per_node <NUM_GPUS> finetuning_math10k_case2.py \
  --llama_model_path <TARGET_FOLDER>/<MODEL_SIZE> \
  --data_path <PATH_TO_DATA_JSON> \
  --output_dir <OUTPUT_DIR> \
  --adapter_layer 30 \
  --adapter_len 10 \
  --max_seq_len 256 \
  --batch_size 16 \
  --epochs 5
```

You can also use other task scripts in `alpaca_finetuning_v1/` (e.g., `finetuning_math401_case2.py`, `finetuning_SVAMP_case2.py`).

### What `--data_path` should look like

Each fine-tuning script expects a task-specific JSON format (it is constructed inside the script into a prompt + answer, then tokenized). Use the corresponding script to determine the required keys/fields for your dataset.

## Citations

If you use this work, please cite the paper:

```bibtex
@misc{wang2024deconfoundedcausalityawareparameterefficientfinetuning,
 title={Deconfounded Causality-aware Parameter-Efficient Fine-Tuning for Problem-Solving Improvement of LLMs}, 
 author={Ruoyu Wang and Xiaoxuan Li and Lina Yao},
 year={2024},
 eprint={2409.02686},
 archivePrefix={arXiv},
 primaryClass={cs.CL},
 url={https://arxiv.org/abs/2409.02686}, 
}
```

This code builds on the open-source LLaMA-Adapter implementation:
[`opengvlab/llama-adapter`](https://github.com/opengvlab/llama-adapter).


