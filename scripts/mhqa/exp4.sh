#!/bin/bash
# exp4: 30% bios + 70% implicit 2-hop RDF
# Pipeline: pretrain → LoRA finetune → eval
set -e

NGPU=4

# ── Step 1: Pretrain ──────────────────────────────────────────────────────────
torchrun --standalone --nproc_per_node=$NGPU train_bios.py --config configs/pretrain/exp4.yaml

# ── Step 2: LoRA finetune ────────────────────────────────────────────────────
LORA_CKPT=$(python3 -c "
import yaml, os
cfg = yaml.safe_load(open('configs/finetuning/exp4.yaml'))
tc = cfg['training']
print(os.path.join(tc['log_dir'], tc['run_name'] + '_lora.pt'))
")

CUDA_VISIBLE_DEVICES=0 python finetune_graph_qa.py --config configs/finetuning/exp4.yaml

# ── Step 3: Eval ──────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python eval_graph_qa.py \
    --config    configs/finetuning/exp4.yaml \
    --lora_ckpt "$LORA_CKPT"
