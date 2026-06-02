"""
Evaluate a LoRA-finetuned Graph-BioS model on 1-hop / 2-hop / 3-hop QA.

Reports two metrics per file, broken down by (hops, attr) and (hops, chain):
  P_train -- first-token accuracy on all training QA files
  P_test  -- exact-match  accuracy on all eval     QA files

Config expects qa_train / qa_val as either a single path or a list of paths.

Usage:
    python eval_graph_qa.py --config configs/finetune_qa_graph_..._1hop2hop.yaml \
        --lora_ckpt logs/qa_graph_.../..._lora.pt
"""

import argparse
import json
import math
import os
from collections import defaultdict

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from model import GPT, GPTConfig

# -- LoRA modules (identical to finetune_graph_qa.py) ---------------------

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

# -- CLI -------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--config",     required=True)
parser.add_argument("--lora_ckpt",  default=None,
                    help="path to LoRA checkpoint (mutually exclusive with --full_ckpt)")
parser.add_argument("--full_ckpt",  default=None,
                    help="path to a full-finetune checkpoint (mutually exclusive with --lora_ckpt)")
parser.add_argument("--debug",      action="store_true")
parser.add_argument("--n_eval",     type=int, default=0,
                    help="evaluate on first N individuals per file (0 = all)")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--max_new",    type=int, default=24)
parser.add_argument("--val_files",  type=str, nargs="+", default=None,
                    help="override qa_val from config (useful for 3-hop eval only)")
parser.add_argument("--mid_files",  type=str, nargs="+", default=None,
                    help="P_train_noFT files evaluated with c_first metric (3-split eval)")
parser.add_argument("--train_extra", type=str, nargs="+", default=None,
                    help="extra files appended to qa_train for eval only (e.g. 3-hop_train_ft)")
args = parser.parse_args()

assert (args.lora_ckpt is None) ^ (args.full_ckpt is None), \
    "exactly one of --lora_ckpt or --full_ckpt must be provided"

with open(args.config) as f:
    cfg = yaml.safe_load(f)

mcfg = cfg["model"]
fcfg = cfg["finetuning"]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

def _as_list(x):
    if x is None:
        return []
    return [x] if isinstance(x, str) else list(x)

qa_train_paths = _as_list(fcfg.get("qa_train")) + (_as_list(args.train_extra) if args.train_extra else [])
qa_val_paths   = _as_list(args.val_files) if args.val_files else _as_list(fcfg.get("qa_val"))
qa_mid_paths   = _as_list(args.mid_files) if args.mid_files else []

# -- Load base model -------------------------------------------------------

model = GPT(GPTConfig(**mcfg))

if args.lora_ckpt is not None:
    # LoRA path: load pretrain weights, inject LoRA modules, then load LoRA delta
    ckpt_base = torch.load(fcfg["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt_base["model"])
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False

    n_embd     = model.config.n_embd
    vocab_size = model.config.vocab_size

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

    lora_ckpt  = torch.load(args.lora_ckpt, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(lora_ckpt["lora"], strict=False)
    if unexpected:
        print(f"WARNING: unexpected keys in lora checkpoint: {unexpected}")
    print(f"loaded LoRA checkpoint: {args.lora_ckpt}")
else:
    # Full-finetune path: the checkpoint already contains the complete updated model state
    full_ckpt = torch.load(args.full_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(full_ckpt["model"])
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    print(f"loaded full checkpoint: {args.full_ckpt}")

model.eval()

# -- Tokeniser -------------------------------------------------------------

enc = tiktoken.get_encoding("gpt2")
EOT = enc._special_tokens["<|endoftext|>"]

# -- Batched greedy decoding ----------------------------------------------

@torch.no_grad()
def greedy_batch(prompt_ids_list, max_new):
    B       = len(prompt_ids_list)
    lengths = {len(p) for p in prompt_ids_list}
    assert len(lengths) == 1, f"non-uniform prompt lengths: {sorted(lengths)}"

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
                if tok >= EOT:  # stop on EOT or any special/graph token (>= 50257)
                    finished[i] = True
                else:
                    generated[i].append(tok)
        if all(finished):
            break
    return generated

# -- Eval helpers ----------------------------------------------------------

ATTRS = ["birthday", "birthcity", "university", "major",
         "employer", "employer_city", "friend", "enemy"]


def load_records(path, n_eval):
    records = []
    seen_ids = set()
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


def _norm(s):
    return " ".join(s.lower().split())


def _prefix_match(gen, gold):
    g = _norm(gen); a = _norm(gold)
    if not a or not g.startswith(a):
        return False
    if len(g) == len(a):
        return True
    return not g[len(a)].isalnum()


def eval_file(path, label):
    records = load_records(path, args.n_eval)
    print(f"\nevaluating {label}: {len(records):,} QA pairs "
          f"(batch={args.batch_size}, max_new={args.max_new})")

    # Counters keyed by (hops, attr) and (hops, chain)
    n_total   = defaultdict(int)
    c_exact   = defaultdict(int)
    c_prefix  = defaultdict(int)
    c_first   = defaultdict(int)
    chain_total = defaultdict(int)
    chain_exact = defaultdict(int)
    chain_first = defaultdict(int)

    # Length-bucket for uniform-length batches (no attn_mask in model)
    records_with_ids = [(r, enc.encode(r["prompt"])) for r in records]
    buckets = defaultdict(list)
    for r, pids in records_with_ids:
        buckets[len(pids)].append((r, pids))

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
                hops      = rec["hops"]
                chain_key = "->".join(rec.get("chain", []) + [attr])

                k = (hops, attr)
                n_total[k]  += 1
                c_exact[k]  += int(_norm(gen_text) == _norm(gold_text))
                c_prefix[k] += int(_prefix_match(gen_text, gold_text))
                gold_first  = enc.encode(" " + gold_text)[0]
                gen_first   = gen_ids[0] if gen_ids else -1
                c_first[k]  += int(gen_first == gold_first)

                ck = (hops, chain_key)
                chain_total[ck] += 1
                chain_exact[ck] += int(_norm(gen_text) == _norm(gold_text))
                chain_first[ck] += int(gen_first == gold_first)

                if args.debug:
                    print(f"\nID {rec['id']}  hops={hops}  chain={rec.get('chain')}  attr={attr}")
                    print(f"PROMPT: {rec['prompt']}")
                    print(f"GOLD:   {gold_text.lower()}")
                    print(f"GEN:    {gen_text.lower()}")
    pbar.close()

    return {
        "n_total":     dict(n_total),
        "c_exact":     dict(c_exact),
        "c_prefix":    dict(c_prefix),
        "c_first":     dict(c_first),
        "chain_total": dict(chain_total),
        "chain_exact": dict(chain_exact),
        "chain_first": dict(chain_first),
    }


def print_hop_breakdown(stats, metric_key, label):
    print(f"\n-- {label} --")
    totals = stats["n_total"]
    correct = stats[metric_key]
    hops_set = sorted({h for h, _ in totals.keys()})
    overall_c = overall_t = 0
    for hops in hops_set:
        print(f"\n  [{hops}-hop]")
        hop_c = hop_t = 0
        for attr in ATTRS:
            k = (hops, attr)
            t = totals.get(k, 0)
            c = correct.get(k, 0)
            if t == 0:
                continue
            print(f"    {attr:<15s} {c:>7d}/{t:<7d}  {c/t:.3f}")
            hop_c += c; hop_t += t
        if hop_t:
            print(f"    {'OVERALL':<15s} {hop_c:>7d}/{hop_t:<7d}  {hop_c/hop_t:.3f}")
        overall_c += hop_c; overall_t += hop_t
    overall = overall_c / overall_t if overall_t else 0.0
    print(f"\n  ALL-HOPS {overall_c}/{overall_t}  {overall:.3f}")
    print("-----------------------------------------------------")
    return overall


def print_chain_breakdown(stats, metric_key, label):
    print(f"\n-- {label} by chain --")
    correct = stats["chain_" + metric_key.replace("c_", "")]
    totals  = stats["chain_total"]
    hops_set = sorted({h for h, _ in totals.keys()})
    for hops in hops_set:
        print(f"  [{hops}-hop]")
        for (h, chain), t in sorted(totals.items()):
            if h != hops:
                continue
            c = correct.get((h, chain), 0)
            print(f"    {chain:<45s} {c:>7d}/{t:<7d}  {c/t:.3f}")


def run_metric(paths, metric_key, label):
    overall = {"n_total": defaultdict(int), "c_exact": defaultdict(int),
               "c_prefix": defaultdict(int), "c_first": defaultdict(int),
               "chain_total": defaultdict(int), "chain_exact": defaultdict(int),
               "chain_first": defaultdict(int)}
    for path in paths:
        stats = eval_file(path, f"{label}:{os.path.basename(path)}")
        # Stream per-file breakdown so partial results are visible while
        # later (larger) files are still running.
        print_hop_breakdown(stats, metric_key, f"{label} [{os.path.basename(path)}]")
        for key in overall.keys():
            for k, v in stats[key].items():
                overall[key][k] += v
    overall = {k: dict(v) for k, v in overall.items()}
    acc = print_hop_breakdown(overall, metric_key, f"{label} [AGGREGATE]")
    print_chain_breakdown(overall, metric_key, label)
    return acc, overall

# -- Run -------------------------------------------------------------------

results = {}
if qa_train_paths:
    p_train, train_stats = run_metric(qa_train_paths, "c_first",
                                      "P_train_FT (first-token)")
    results["p_train"] = train_stats
    results["p_train_acc"] = p_train
else:
    p_train = None

if qa_mid_paths:
    p_mid, mid_stats = run_metric(qa_mid_paths, "c_first",
                                  "P_train_noFT (first-token)")
    results["p_train_noFT"] = mid_stats
    results["p_train_noFT_acc"] = p_mid
else:
    p_mid = None

if qa_val_paths:
    p_test, val_stats = run_metric(qa_val_paths, "c_exact",
                                   "P_test (exact-match)")
    results["p_test"] = val_stats
    results["p_test_acc"] = p_test
else:
    p_test = None

summary = []
if p_train is not None: summary.append(f"P_train_FT={p_train:.3f}")
if p_mid   is not None: summary.append(f"P_train_noFT={p_mid:.3f}")
if p_test  is not None: summary.append(f"P_test={p_test:.3f}")
print("\nSUMMARY  " + "  ".join(summary))

# -- Save ------------------------------------------------------------------

def _jsonable(obj):
    """Recursively turn tuple keys into 'hops|attr' strings for JSON dumping."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            key = "|".join(str(x) for x in k) if isinstance(k, tuple) else k
            out[key] = _jsonable(v)
        return out
    if isinstance(obj, list):
        return [_jsonable(x) for x in obj]
    return obj

suffix   = "_debug" if args.debug else ""
ckpt_path_for_out = args.lora_ckpt if args.lora_ckpt else args.full_ckpt
out_path = os.path.join(os.path.dirname(ckpt_path_for_out),
                        f"eval_results_graph{suffix}.json")
with open(out_path, "w") as f:
    json.dump(_jsonable(results), f, indent=2)
print(f"results saved -> {out_path}")
