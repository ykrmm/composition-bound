"""
LoRA QA fine-tuning on a pretrained BioS GPT-2+RoPE checkpoint.

"""

import argparse
import json
import math
import os
import random
import time

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb
import yaml

from model import GPT, GPTConfig

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

mcfg = cfg["model"]
fcfg = cfg["finetuning"]
wcfg = cfg["wandb"]

device      = "cuda" if torch.cuda.is_available() else "cpu"
device_type = "cuda" if device.startswith("cuda") else "cpu"
print(f"device: {device}")

# ── Hyperparams ───────────────────────────────────────────────────────────────

max_lr       = fcfg["max_lr"]
weight_decay = fcfg["weight_decay"]
eps          = fcfg["eps"]
max_steps    = fcfg["max_steps"]
batch_size   = fcfg["batch_size"]
r_q          = fcfg["lora_r_q"]
r_v          = fcfg["lora_r_v"]
r_emb        = fcfg["lora_r_emb"]
log_dir      = fcfg["log_dir"]
run_name     = fcfg["run_name"]
qa_train         = fcfg["qa_train"]
checkpoint       = fcfg["checkpoint"]
first_token_only = fcfg.get("first_token_only", False)
lr_schedule      = fcfg.get("lr_schedule", "cosine")   # "cosine" or "linear"

min_lr = max_lr * 0.1

def get_lr(it):
    if it >= max_steps:
        return 0.0 if lr_schedule == "linear" else min_lr
    if lr_schedule == "linear":
        return max_lr * (1.0 - it / max_steps)
    # cosine decay to 10% of max_lr
    coeff = 0.5 * (1.0 + math.cos(math.pi * it / max_steps))
    return min_lr + coeff * (max_lr - min_lr)

# ── LoRA modules ──────────────────────────────────────────────────────────────

class LoRAMergedLinear(nn.Module):
    """Replaces c_attn. Applies LoRA to Q and V slices; K is unchanged.

    alpha = r  →  scaling = 1.0  (standard convention, same as no scaling).
    """
    def __init__(self, in_features, out_features, r_q, r_v):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        n = out_features // 3  # n_embd

        self.lora_A_q = nn.Linear(in_features, r_q, bias=False)
        self.lora_B_q = nn.Linear(r_q, n, bias=False)

        self.lora_A_v = nn.Linear(in_features, r_v, bias=False)
        self.lora_B_v = nn.Linear(r_v, n, bias=False)

        nn.init.kaiming_uniform_(self.lora_A_q.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_q.weight)
        nn.init.kaiming_uniform_(self.lora_A_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_v.weight)

    def forward(self, x):
        qkv = self.linear(x)
        n   = qkv.shape[-1] // 3
        q, k, v = qkv.split(n, dim=-1)
        q = q + self.lora_B_q(self.lora_A_q(x))
        v = v + self.lora_B_v(self.lora_A_v(x))
        return torch.cat([q, k, v], dim=-1)


class LoRAEmbedding(nn.Module):
    """Replaces wte. Adds a low-rank delta to token embeddings.

    alpha = r  →  scaling = 1.0.
    """
    def __init__(self, num_embeddings, embedding_dim, r):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lora_A    = nn.Embedding(num_embeddings, r)
        self.lora_B    = nn.Linear(r, embedding_dim, bias=False)

        # PEFT convention for embeddings: A=zeros, B=random
        # → A (sparse lookup) learns first, one gradient per seen token
        # (opposite of linear LoRA where B=zeros, A=random)
        nn.init.zeros_(self.lora_A.weight)
        nn.init.normal_(self.lora_B.weight, std=0.02)

    def forward(self, idx):
        return self.embedding(idx) + self.lora_B(self.lora_A(idx))

# ── Load base model ───────────────────────────────────────────────────────────

torch.manual_seed(42)
torch.set_float32_matmul_precision("high")

ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
model = GPT(GPTConfig(**mcfg))
model.load_state_dict(ckpt["model"])
model.to(device)
print(f"loaded checkpoint: {checkpoint}  (step {ckpt.get('step', '?')})")

# Freeze all base parameters
for p in model.parameters():
    p.requires_grad = False

n_embd     = model.config.n_embd
vocab_size = model.config.vocab_size

# ── Inject LoRA by replacing modules in-place ─────────────────────────────────

# Replace c_attn in every attention block
for block in model.transformer.h:
    orig          = block.attn.c_attn          # original nn.Linear (frozen)
    lora_layer    = LoRAMergedLinear(n_embd, 3 * n_embd, r_q, r_v).to(device)
    lora_layer.linear.weight = orig.weight     # reuse frozen pretrained weight
    lora_layer.linear.bias   = orig.bias       # reuse frozen pretrained bias
    block.attn.c_attn = lora_layer

# Replace wte with LoRAEmbedding
orig_wte         = model.transformer.wte
lora_wte         = LoRAEmbedding(vocab_size, n_embd, r_emb).to(device)
lora_wte.embedding.weight = orig_wte.weight   # reuse frozen pretrained embedding
model.transformer.wte = lora_wte

# ── Optimizer (only trainable = LoRA params) ──────────────────────────────────

lora_params = [p for p in model.parameters() if p.requires_grad]
n_params    = sum(p.numel() for p in lora_params)
print(f"LoRA trainable parameters: {n_params:,}")

optimizer = torch.optim.AdamW(
    lora_params,
    lr=max_lr,
    betas=(0.9, 0.95),
    weight_decay=weight_decay,
    eps=eps,
)

# ── QA dataloader ─────────────────────────────────────────────────────────────

enc = tiktoken.get_encoding("gpt2")
EOT = enc._special_tokens["<|endoftext|>"]
PAD = EOT

class QADataLoader:
    """Loads qa_train.jsonl, tokenises, returns padded batches with loss mask.

    Shuffles at the individual level (all 6 attributes of each person stay
    together) so that each batch gradient is coherent for a fixed set of
    individuals rather than averaging over many different ones.
    batch_size must be a multiple of n_attrs (6).
    """
    def __init__(self, path, batch_size):
        self.batch_size = batch_size

        # Group samples by individual id
        by_id: dict[int, list] = {}
        with open(path) as f:
            for line in f:
                rec    = json.loads(line)
                prompt = enc.encode(rec["prompt"])
                answer = enc.encode(" " + rec["answer"])
                tokens = prompt + answer + [EOT]
                if first_token_only:
                    mask = [0] * len(prompt) + [1] + [0] * len(answer)
                else:
                    mask = [0] * len(prompt) + [1] * len(answer) + [1]
                by_id.setdefault(rec["id"], []).append((tokens, mask))

        # Each "group" is the list of (tokens, mask) for one individual
        self.groups = list(by_id.values())
        random.shuffle(self.groups)

        # Flatten groups into samples (groups stay contiguous)
        self.samples = [s for g in self.groups for s in g]
        self.n_attrs = len(self.groups[0])   # typically 6
        self.pos     = 0
        print(f"loaded {len(self.samples):,} QA samples "
              f"({len(self.groups):,} individuals × {self.n_attrs} attrs) from {path}")

    def next_batch(self):
        B   = self.batch_size
        end = self.pos + B
        if end > len(self.samples):
            # Re-shuffle at individual level and rebuild flat list
            random.shuffle(self.groups)
            self.samples = [s for g in self.groups for s in g]
            self.pos = 0
            end      = B

        batch = self.samples[self.pos : end]
        self.pos = end

        max_len = max(len(t) for t, _ in batch)
        xs, ys, ms = [], [], []
        for tokens, mask in batch:
            pad = max_len - len(tokens)
            t   = tokens + [PAD] * pad
            m   = mask   + [0]   * pad
            xs.append(t[:-1])
            ys.append(t[1:])
            ms.append(m[1:])   # mask aligns with targets (shifted by 1)

        x = torch.tensor(xs, dtype=torch.long)
        y = torch.tensor(ys, dtype=torch.long)
        m = torch.tensor(ms, dtype=torch.float)
        return x, y, m


loader = QADataLoader(qa_train, batch_size)

# ── wandb ─────────────────────────────────────────────────────────────────────

os.makedirs(log_dir, exist_ok=True)
with open(os.path.join(log_dir, "config.yaml"), "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)
run = wandb.init(
    entity=wcfg["entity"],
    project=wcfg["project"],
    name=run_name,
    config={**fcfg, "lora_params": n_params},
)

# ── Training loop ─────────────────────────────────────────────────────────────

model.train()
pbar = tqdm.tqdm(total=max_steps, desc=run_name)
st   = time.time()

for step in range(max_steps):
    x, y, mask = loader.next_batch()
    x, y, mask = x.to(device), y.to(device), mask.to(device)

    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        logits, _ = model(x)   # GPT.forward unchanged; loss computed here with mask
        loss_raw  = F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), reduction="none"
        )
        loss = (loss_raw * mask.view(-1)).sum() / mask.sum().clamp(min=1)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    optimizer.step()

    run.log({"train_loss": loss.item(), "lr": lr}, step=step)
    pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.2e}"})
    pbar.update(1)

pbar.close()

# ── Save LoRA checkpoint ──────────────────────────────────────────────────────

# Save only the trainable (LoRA) parameters, identified by requires_grad
trainable_keys = {name for name, p in model.named_parameters() if p.requires_grad}
lora_state     = {k: v.cpu() for k, v in model.state_dict().items()
                  if k in trainable_keys}

ckpt_path = os.path.join(log_dir, f"{run_name}_lora.pt")
torch.save({
    "lora":      lora_state,
    "config":    mcfg,
    "ft_config": fcfg,
    "step":      max_steps,
}, ckpt_path)
print(f"LoRA checkpoint saved → {ckpt_path}")

elapsed = int(time.time() - st)
run.summary["total_time"] = f"{elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}"
run.finish()
