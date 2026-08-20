# [EMNLP 2026 Submission] Multi-Hop Knowledge Composition is Bound by Pretraining Exposure

This repository contains the code for the submitted EMNLP 2026 paper "Multi-Hop Knowledge Composition is Bound by Pretraining Exposure". It extends the synthetic biography framework (Phy. LLM 3.1) with inter-individual relations (friend, enemy) and multi-hop QA. It presents also the 9 pretraining augmentation strategies to study compositional generalization.


<img width="887" height="471" alt="Capture d’écran 2026-08-20 à 23 25 37" src="https://github.com/user-attachments/assets/f6e01a51-cd46-4d7c-9110-cd52da030218" />

---

## Setup

```bash
pip install torch==2.8.0 tiktoken numpy tqdm wandb pyyaml
```

Experiments were run on 4×H100 GPUs. Pretraining takes ~8 hours, LoRA finetuning ~2 hours per experiment.

---

## Reproducing the experiments

### 1. Generate data

```bash
bash scripts/data/gen_phyllm_data.sh
```

This script generates all data needed for experiments 1–9:

- `graph_bios_data/individuals.json` — 100K individuals with 6 attributes + friend/enemy relations
- `graph_bios_data/bios_multi5p_fullname_all.txt` — 1-hop NL biographies (5 permuted reps per individual)
- `graph_bios_data/bioG_2hop_nl_{implicit,explicit}_50k_all.txt` — 2-hop NL augmentation for P_comp
- `graph_bios_data/bioG_2hop_triple_{implicit,explicit}_50k_all.txt` — 2-hop RDF augmentation for P_comp
- Mixed pretrain corpora for each experiment (tokenized as `.npy`)
- `graph_qa_data/` — 1-hop, 2-hop, 3-hop QA pairs

**Population split.** The 100K individuals are split 50/50. P_comp (ids 0–49999) receives compositional augmentation during pretraining. P_held (ids 50000–99999) is restricted to atomic 1-hop biographies only; it never appears as a bridge entity in any compositional chain.

### 2. Pretrain

```bash
torchrun --standalone --nproc_per_node=4 train_bios.py --config configs/pretrain/expN.yaml
```

Replace `N` with the experiment number (2–9; see table below). The baseline (Exp 0) uses `configs/pretrain/baseline_phy_llm.yaml`.

All models are GPT-2 Small (124M, 12L/12H/768D) with rotary positional embeddings, trained from scratch. Training: 800K steps, batch 49152 tokens, cosine LR from 1e-3 to 1e-4, 1K warmup steps.

### 3. LoRA finetune

```bash
python finetune_graph_qa.py --config configs/finetuning/expN.yaml
```

LoRA applied to query, value, and embedding layers. Training: 150K steps, batch 48, lr 3e-4. The finetuning config for each experiment contains the best (r_qv, r_emb) pair from the sweep reported in the paper (Table 13).

### 4. Evaluate

```bash
python eval_graph_qa.py --config configs/finetuning/expN.yaml --lora_ckpt <path_to_lora.pt>
```

Reports first-token accuracy (P_comp) and exact-match accuracy (P_held) for 1-hop, 2-hop, and 3-hop queries.

---

## Experiment configurations

| Exp | Augmentation | NL | RDF | Implicit | Explicit | Vocab |
|-----|--------------|----|-----|----------|----------|-------|
| 0   | Baseline (no augmentation) | — | — | — | — | 50257 |
| 1   | 1-hop RDF | — | ✓ | — | — | 50304 |
| 2   | 2-hop implicit NL | ✓ | — | ✓ | — | 50257 |
| 3   | 2-hop explicit NL | ✓ | — | — | ✓ | 50257 |
| 4   | 2-hop implicit RDF | — | ✓ | ✓ | — | 50304 |
| 5   | 2-hop explicit RDF | — | ✓ | — | ✓ | 50304 |
| 6   | 2-hop implicit + explicit RDF | — | ✓ | ✓ | ✓ | 50304 |
| 7   | 2-hop implicit + explicit NL | ✓ | — | ✓ | ✓ | 50257 |
| 8   | 2-hop implicit + explicit NL + RDF | ✓ | ✓ | ✓ | ✓ | 50304 |
| 9   | All formats (incl. 1-hop RDF) | ✓ | ✓ | ✓ | ✓ | 50304 |

Pretraining mix ratios follow Table 10 in the paper. All conditions include 1-hop NL biographies (multi5p-permute) for all 100K individuals.

---

## Repository structure

```
.
├── model.py                    # GPT-2 with RoPE
├── train_bios.py               # Distributed pretraining
├── finetune_qa.py              # LoRA finetuning (baseline / phy-llm)
├── finetune_graph_qa.py        # LoRA finetuning (multi-hop, multi-file QA)
├── eval_qa.py                  # Evaluation (baseline)
├── eval_graph_qa.py            # Evaluation (multi-hop, per-hop breakdown)
├── gen_data/
│   ├── generate_graph_bios.py  # 1-hop NL biographies with relations
│   ├── generate_bioG_2hop_nl.py    # 2-hop NL augmentation
│   ├── generate_bioG_2hop_triple.py # 2-hop RDF augmentation
│   ├── generate_graph_qa.py    # 1/2/3-hop QA pairs
│   ├── mix_pretrain.py         # Combine corpora at given ratios
│   ├── tokenizer_graph_bios.py # GPT-2 tokenizer + [ENTITY]/[RELATION]/[VALUE] tokens
│   └── bioG_common.py          # Shared constants and utilities
├── configs/
│   ├── pretrain/               # exp0 (baseline) + exp2–exp9
│   └── finetuning/             # exp2–exp9 (LoRA configs)
└── scripts/
    ├── data/                   # Data generation scripts
    ├── mhqa/                   # End-to-end pipeline scripts (pretrain → finetune → eval)
    └── phy_llm/                # Baseline (1-hop) pipeline scripts
```

---

## Notes on vocabulary size

Experiments using RDF triples (Exp 1, 4, 5, 6, 8, 9) extend the GPT-2 vocabulary with five special tokens — `[ENTITY]` (50257), `[RELATION]` (50258), `[VALUE]` (50259), `[LINKED]` (50260), `[END_ENTITY]` (50261) — padded to 50304 (multiple of 64 for tensor core alignment). NL-only experiments use the standard GPT-2 vocabulary (50257).
