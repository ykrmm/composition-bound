"""
Evaluate a LoRA-finetuned BioS model on both paper metrics in one pass:

"""

import argparse
import json
import math
import os

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from model import GPT, GPTConfig

# ── LoRA modules (identical to finetune_qa.py) ────────────────────────────────

class LoRAMergedLinear(nn.Module):
    def __init__(self, in_features, out_features, r_q, r_v):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        n = out_features // 3

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
    def __init__(self, num_embeddings, embedding_dim, r):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lora_A    = nn.Embedding(num_embeddings, r)
        self.lora_B    = nn.Linear(r, embedding_dim, bias=False)

        nn.init.zeros_(self.lora_A.weight)
        nn.init.normal_(self.lora_B.weight, std=0.02)

    def forward(self, idx):
        return self.embedding(idx) + self.lora_B(self.lora_A(idx))

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--config",     required=True)
parser.add_argument("--lora_ckpt",  required=True)
parser.add_argument("--debug",      action="store_true")
parser.add_argument("--n_eval",     type=int, default=0,
                    help="evaluate on first N individuals (0 = all)")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--max_new",    type=int, default=24)
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

mcfg = cfg["model"]
fcfg = cfg["finetuning"]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")
print(f"debug mode: {args.debug}")

# ── Load base model ───────────────────────────────────────────────────────────

ckpt_base = torch.load(fcfg["checkpoint"], map_location="cpu", weights_only=False)
model = GPT(GPTConfig(**mcfg))
model.load_state_dict(ckpt_base["model"])
model.to(device)

for p in model.parameters():
    p.requires_grad = False

n_embd     = model.config.n_embd
vocab_size = model.config.vocab_size

# ── Inject LoRA modules (same structure as finetune_qa.py) ────────────────────

for block in model.transformer.h:
    orig       = block.attn.c_attn
    lora_layer = LoRAMergedLinear(n_embd, 3 * n_embd,
                                  fcfg["lora_r_q"], fcfg["lora_r_v"]).to(device)
    lora_layer.linear.weight = orig.weight
    lora_layer.linear.bias   = orig.bias
    block.attn.c_attn = lora_layer

orig_wte         = model.transformer.wte
lora_wte         = LoRAEmbedding(vocab_size, n_embd, fcfg["lora_r_emb"]).to(device)
lora_wte.embedding.weight = orig_wte.weight
model.transformer.wte = lora_wte

# ── Load LoRA weights ─────────────────────────────────────────────────────────

lora_ckpt  = torch.load(args.lora_ckpt, map_location=device, weights_only=False)
lora_state = lora_ckpt["lora"]
missing, unexpected = model.load_state_dict(lora_state, strict=False)
if unexpected:
    print(f"WARNING: unexpected keys in lora checkpoint: {unexpected}")

model.eval()
print(f"loaded LoRA checkpoint: {args.lora_ckpt}")

# ── Tokeniser ─────────────────────────────────────────────────────────────────

enc = tiktoken.get_encoding("gpt2")
EOT = enc._special_tokens["<|endoftext|>"]

# ── Batched greedy decoding ───────────────────────────────────────────────────

@torch.no_grad()
def greedy_batch(prompt_ids_list: list[list[int]], max_new: int) -> list[list[int]]:
    """Greedy decoding on a batch of prompts.

    Requires all prompts in the batch to have the same length — the model has
    no attention_mask path and uses is_causal=True, so any left-padding with
    EOT would be attended to by real prompt tokens and pollute their hidden
    states. Callers must length-bucket before calling this.
    """
    B          = len(prompt_ids_list)
    lengths    = {len(p) for p in prompt_ids_list}
    assert len(lengths) == 1, (
        f"greedy_batch requires uniform prompt length; got {sorted(lengths)}"
    )

    x = torch.tensor(prompt_ids_list, dtype=torch.long, device=device)

    generated = [[] for _ in range(B)]
    finished  = [False] * B

    for _ in range(max_new):
        logits, _ = model(x[:, -model.config.block_size:])
        next_tok  = logits[:, -1, :].argmax(dim=-1)
        x         = torch.cat([x, next_tok.unsqueeze(1)], dim=1)

        for i in range(B):
            if not finished[i]:
                tok = next_tok[i].item()
                if tok == EOT:
                    finished[i] = True
                else:
                    generated[i].append(tok)

        if all(finished):
            break

    return generated

# ── Evaluation helper ─────────────────────────────────────────────────────────

ATTRS = ["birthday", "birthcity", "university", "major", "employer", "employer_city"]


def load_records(path: str, n_eval: int) -> list[dict]:
    records: list[dict] = []
    seen_ids: set[int]  = set()
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if n_eval > 0:
                if rec["id"] not in seen_ids:
                    if len(seen_ids) >= n_eval:
                        continue
                    seen_ids.add(rec["id"])
            records.append(rec)
    return records


def _norm(s: str) -> str:
    """Lowercase + collapse whitespace, for comparing gen and gold."""
    return " ".join(s.lower().split())


def _prefix_match(gen: str, gold: str) -> bool:
    """True if gen (after norm) starts with gold (after norm) at a word boundary.

    The model overfits to bio-style continuation and emits the answer followed by
    extra text (e.g. ' marriott. matthew makenzie ...'). Strict equality misses
    these. Prefix-match with a non-alnum boundary handles this without false
    positives like 'atlanta, ga' matching 'atlanta, gas'.
    """
    g = _norm(gen)
    a = _norm(gold)
    if not a or not g.startswith(a):
        return False
    if len(g) == len(a):
        return True
    return not g[len(a)].isalnum()


def eval_split(path: str, label: str) -> dict:
    records = load_records(path, args.n_eval)
    print(f"\nevaluating {label}: {len(records):,} QA pairs  "
          f"(batch={args.batch_size}, max_new={args.max_new})")

    correct_exact  = {a: 0 for a in ATTRS}
    correct_prefix = {a: 0 for a in ATTRS}
    correct_first  = {a: 0 for a in ATTRS}
    total          = {a: 0 for a in ATTRS}

    records_with_ids = [(r, enc.encode(r["prompt"])) for r in records]
    buckets: dict[int, list] = {}
    for r, pids in records_with_ids:
        buckets.setdefault(len(pids), []).append((r, pids))

    total_batches = sum(
        (len(items) + args.batch_size - 1) // args.batch_size
        for items in buckets.values()
    )
    pbar = tqdm(total=total_batches, desc=label)

    for length, items in buckets.items():
        for i in range(0, len(items), args.batch_size):
            batch_items  = items[i : i + args.batch_size]
            batch        = [r for r, _ in batch_items]
            prompt_ids_l = [pids for _, pids in batch_items]
            gen_ids_l    = greedy_batch(prompt_ids_l, max_new=args.max_new)
            pbar.update(1)

            for rec, gen_ids in zip(batch, gen_ids_l):
                gen_text  = enc.decode(gen_ids).strip()
                gold_text = rec["answer"].strip()
                attr      = rec["attr"]
                total[attr] += 1

                correct_exact[attr]  += int(_norm(gen_text) == _norm(gold_text))
                correct_prefix[attr] += int(_prefix_match(gen_text, gold_text))

                gold_first = enc.encode(" " + gold_text)[0]
                gen_first  = gen_ids[0] if gen_ids else -1
                correct_first[attr] += int(gen_first == gold_first)

                if args.debug:
                    print(f"\nID {rec['id']}  ATTR: {attr}")
                    print(f"PROMPT:\n{rec['prompt']}")
                    print(f"GOLD: {gold_text.lower()}")
                    print(f"GEN:  {gen_text.lower()}")
    pbar.close()

    return {
        "correct_exact":  correct_exact,
        "correct_prefix": correct_prefix,
        "correct_first":  correct_first,
        "total":          total,
    }


def print_results(stats: dict, metric_key: str) -> float:
    correct = stats[metric_key]
    total   = stats["total"]
    total_c = total_t = 0
    for attr in ATTRS:
        t = total[attr]
        c = correct[attr]
        print(f"  {attr:<15s}  {c:>6d}/{t:<6d}  {c/t if t else 0:.3f}")
        total_c += c
        total_t += t
    overall = total_c / total_t if total_t else 0.0
    print(f"  {'OVERALL':<15s}  {total_c:>6d}/{total_t:<6d}  {overall:.3f}")
    print("─────────────────────────────────────────────────")
    return overall

# ── Run both splits ───────────────────────────────────────────────────────────

train_stats = eval_split(fcfg["qa_train"], "P_train (qa_train)")
print("\n── P_train: first-token accuracy (qa_train) ─────")
p_train = print_results(train_stats, "correct_first")

val_stats = eval_split(fcfg["qa_val"], "P_test (qa_val)")
print("\n── P_test: exact-match accuracy (qa_val) ────────")
p_test = print_results(val_stats, "correct_exact")

print(f"\nSUMMARY  P_train={p_train:.3f}  P_test={p_test:.3f}")

# ── Save results ──────────────────────────────────────────────────────────────

def build_result_dict(stats):
    d = {}
    for attr in ATTRS:
        t = stats["total"][attr]
        d[attr] = {
            "correct_exact":  stats["correct_exact"][attr],
            "correct_prefix": stats["correct_prefix"][attr],
            "correct_first":  stats["correct_first"][attr],
            "total": t,
            "acc_exact":  stats["correct_exact"][attr]  / t if t else 0.0,
            "acc_prefix": stats["correct_prefix"][attr] / t if t else 0.0,
            "acc_first":  stats["correct_first"][attr]  / t if t else 0.0,
        }
    te = sum(stats["correct_exact"][a]  for a in ATTRS)
    tp = sum(stats["correct_prefix"][a] for a in ATTRS)
    tf = sum(stats["correct_first"][a]  for a in ATTRS)
    tt = sum(stats["total"][a]          for a in ATTRS)
    d["overall"] = {
        "correct_exact": te, "correct_prefix": tp, "correct_first": tf, "total": tt,
        "acc_exact":  te / tt if tt else 0.0,
        "acc_prefix": tp / tt if tt else 0.0,
        "acc_first":  tf / tt if tt else 0.0,
    }
    return d

suffix   = "_debug" if args.debug else ""
out_path = os.path.join(os.path.dirname(args.lora_ckpt), f"eval_results{suffix}.json")
results  = {
    "p_train": build_result_dict(train_stats),
    "p_test":  build_result_dict(val_stats),
}
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"results saved → {out_path}")
