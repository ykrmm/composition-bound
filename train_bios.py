"""
Train GPT2+RoPE on BioS dataset — Allen-Zhu & Li Physics of LLMs replication 

"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import tqdm
import wandb
import yaml
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, GPTConfig

# ── config ────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

mcfg = cfg["model"]
tcfg = cfg["training"]
ecfg = cfg["eval"]
wcfg = cfg["wandb"]

# ── DDP setup ─────────────────────────────────────────────────────────────────

ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "DDP requires CUDA"
    init_process_group(backend="nccl")
    ddp_rank       = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device         = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

device_type = "cuda" if device.startswith("cuda") else "cpu"
if master_process:
    print(f"using device: {device}  |  world_size: {ddp_world_size}")

# ── hyperparams ───────────────────────────────────────────────────────────────

B                = tcfg["batch_size"]       # per GPU
T                = tcfg["seq_len"]          # 512
max_lr           = tcfg["max_lr"]           # 1e-3
min_lr           = tcfg["min_lr"]           # 1e-4
weight_decay     = tcfg["weight_decay"]     # 0.1
eps              = tcfg["eps"]              # 1e-6
max_steps        = tcfg["max_steps"]        # 80_000
warmup_steps     = tcfg["warmup_steps"]     # 1_000
grad_clip        = tcfg["grad_clip"]        # 1.0
use_compile      = tcfg["use_compile"]
log_dir          = tcfg["log_dir"]
run_name         = tcfg["run_name"]
total_batch_size = tcfg["total_batch_size"] # 96 * 512 = 49_152

assert total_batch_size % (B * T * ddp_world_size) == 0, \
    f"total_batch_size {total_batch_size} must be divisible by B*T*world_size = {B*T*ddp_world_size}"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print(f"total_batch_size: {total_batch_size}  |  grad_accum_steps: {grad_accum_steps}")
    print(f"max_steps: {max_steps}  |  warmup_steps: {warmup_steps}")

# ── lr schedule (cosine with linear warmup — Allen-Zhu) ───────────────────────

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# ── BioS dataloader ───────────────────────────────────────────────────────────

class DataLoaderBioS:
    """Simple dataloader for a single flat uint16 .npy token array.
    Cycles indefinitely; each GPU reads a non-overlapping slice.
    """
    def __init__(self, B, T, process_rank, num_processes, split, tokens_dir, data_variant, train_split="train"):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        # train_split allows using "all" (100K individuals) vs "train" (50K only)
        actual_split = train_split if split == "train" else "val"
        fname = f"{data_variant}_{actual_split}.npy"
        path  = os.path.join(tokens_dir, fname)
        assert os.path.exists(path), f"token file not found: {path}"

        raw = np.load(path).astype(np.int32)
        self.tokens = torch.tensor(raw, dtype=torch.long)
        if master_process:
            print(f"[{split}] loaded {len(self.tokens):,} tokens from {fname}")

        self.reset()

    def reset(self):
        self.pos = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.pos : self.pos + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.pos += B * T * self.num_processes
        if self.pos + B * T * self.num_processes + 1 > len(self.tokens):
            self.reset()
        return x, y

# ── data ──────────────────────────────────────────────────────────────────────

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.set_float32_matmul_precision("high")

data_variant = tcfg["data_variant"]
tokens_dir   = tcfg["tokens_dir"]
train_split  = tcfg.get("train_split", "train")  # "all" = 100K individuals, "train" = 50K

train_loader = DataLoaderBioS(B, T, ddp_rank, ddp_world_size, "train", tokens_dir, data_variant, train_split)
# No val split during pretraining — all 100K individuals are training data (Allen-Zhu setup)
# val_loader only created if val_interval > 0
val_interval = ecfg.get("val_interval", 0)
val_loader   = DataLoaderBioS(B, T, ddp_rank, ddp_world_size, "val", tokens_dir, data_variant) if val_interval > 0 else None

# ── model ─────────────────────────────────────────────────────────────────────

model = GPT(GPTConfig(**mcfg))
model.to(device)
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

# AdamW with Allen-Zhu eps (1e-6) — override configure_optimizers default (1e-8)
optimizer = raw_model.configure_optimizers(weight_decay, max_lr, device_type)
for pg in optimizer.param_groups:
    pg["eps"] = eps

if master_process:
    total_params = sum(p.numel() for p in raw_model.parameters())
    print(f"model params: {total_params:,}")

# ── wandb ─────────────────────────────────────────────────────────────────────

if master_process:
    run = wandb.init(
        entity=wcfg["entity"],
        project=wcfg["project"],
        name=run_name,
        config={**mcfg, **tcfg, "dataset": wcfg["dataset"], "world_size": ddp_world_size},
    )
    os.makedirs(log_dir, exist_ok=True)
    import yaml
    with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

# ── training loop ─────────────────────────────────────────────────────────────

st = time.time()
val_loss_display = None
pbar = tqdm.tqdm(total=max_steps, desc=run_name) if master_process else None

for step in range(max_steps):
    last_step = (step == max_steps - 1)

    # ── validation ────────────────────────────────────────────────────────────
    if val_loader is not None and (step % val_interval == 0 or last_step):
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            for _ in range(ecfg["val_steps"]):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    _, loss = model(x, y)
                val_loss_accum += loss.detach() / ecfg["val_steps"]
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            val_loss_display = val_loss_accum.item()
            run.log({"val_loss": val_loss_display}, step=step)

    # ── checkpoint ────────────────────────────────────────────────────────────
    if master_process and step > 0 and (step % ecfg["checkpoint_interval"] == 0 or last_step):
        ckpt_path = os.path.join(log_dir, f"{run_name}_{step:06d}.pt")
        torch.save({
            "model":    raw_model.state_dict(),
            "config":   raw_model.config,
            "step":     step,
            "val_loss": val_loss_display,
        }, ckpt_path)
        if master_process:
            print(f"checkpoint saved → {ckpt_path}")

    # ── train step ────────────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    optimizer.step()

    if master_process:
        run.log({"train_loss": loss_accum.item(), "norm": norm, "lr": lr}, step=step)
    if pbar is not None:
        postfix = {"loss": f"{loss_accum.item():.4f}", "lr": f"{lr:.2e}"}
        if val_loss_display is not None:
            postfix["val"] = f"{val_loss_display:.4f}"
        pbar.set_postfix(postfix)
        pbar.update(1)

# ── teardown ──────────────────────────────────────────────────────────────────

if pbar is not None:
    pbar.close()

if master_process:
    elapsed = int(time.time() - st)
    elapsed_str = f"{elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}"
    run.summary["total_time"] = elapsed_str
    print(f"Total training time: {elapsed_str}  ({elapsed}s)")
    run.finish()

if ddp:
    destroy_process_group()
