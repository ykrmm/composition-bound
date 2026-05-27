
import argparse
import json
import numpy as np
import torch
from model import GPT, GPTConfig
import tiktoken

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",   required=True)
parser.add_argument("--tokens", required=True, help="Path to pretrain .npy token array")
parser.add_argument("--individuals", default="bios_data/individuals.json",
                    help="Ground-truth individuals JSON used to build the corpus")
parser.add_argument("--n", type=int, default=5,
                    help="Number of individuals to test (taken from the start of the JSON)")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
print(f"loaded checkpoint: {args.ckpt}  (step {ckpt.get('step', '?')})")

model = GPT(GPTConfig(**{
    "block_size": 512, "vocab_size": 50257,
    "n_layer": 12, "n_head": 12, "n_embd": 768, "rope_base": 10000,
}))
model.load_state_dict(ckpt["model"])
model.to(device)
model.eval()

enc = tiktoken.get_encoding("gpt2")
EOT = enc._special_tokens["<|endoftext|>"]

tokens = np.load(args.tokens).astype(np.int64)
print(f"loaded {len(tokens):,} training tokens from {args.tokens}")

# ── Load ground-truth individuals ─────────────────────────────────────────────

with open(args.individuals) as f:
    all_individuals = json.load(f)
print(f"loaded {len(all_individuals):,} individuals from {args.individuals}")

individuals = all_individuals[: args.n]

def full_name(p: dict) -> str:
    return f"{p['first_name']} {p['middle_name']} {p['last_name']}"

def gold_summary(p: dict) -> str:
    return (f"{p['birthmonth'].lower()} {p['birthday']}, {p['birthyear']} "
            f"| birthcity: {p['birthcity'].lower()} "
            f"| university: {p['university'].lower()} "
            f"| major: {p['field'].lower()} "
            f"| employer: {p['company1name'].lower()} "
            f"| employer_city: {p['company1city'].lower()}")

NAMES = [(full_name(p), gold_summary(p)) for p in individuals]

# All 6 bio-style prompts per attribute (what appears in the training data)
# Template: how the fact is introduced in the bio with the full name
ATTR_PROMPTS = [
    ("birthday",     lambda name: f" {name} was born on"),
    ("birthcity",    lambda name: f" {name} was born in"),
    ("university",   lambda name: f" {name} studied at"),
    ("major",        lambda name: f" {name} studied"),
    ("employer",     lambda name: f" {name} worked at"),
    ("employer_city",lambda name: f" {name} worked in"),
]


def greedy_continue(model, prompt_ids, device, max_new=30):
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(max_new):
            logits, _ = model(x[:, -512:])
            next_tok = logits[0, -1, :].argmax().item()
            if next_tok == EOT:
                break
            x = torch.cat([x, torch.tensor([[next_tok]], device=device)], dim=1)
    return enc.decode(x[0, len(prompt_ids):].tolist()).strip()


def find_name_in_tokens(name, tokens, enc):
    """Search for name in token array. Tries with and without leading space
    (individual 0 starts without space; all others are preceded by space after EOT)."""
    for prefix in [" " + name, name]:
        ids = enc.encode(prefix)
        n = len(ids)
        for i in range(len(tokens) - n):
            if list(tokens[i:i + n]) == ids:
                return i, prefix
    return -1, None


# ── TEST 1: short prompt, all 6 attributes ────────────────────────────────────

print("\n" + "=" * 60)
print("TEST 1: BIO-STYLE PROMPTS — all 6 attributes (name only, no context)")
print("Tests if the pretrained model can recall facts from the name alone.")
print("=" * 60)

for name, gold in NAMES:
    print(f"\nNAME: {name}   GOLD: {gold}")
    for attr, pfn in ATTR_PROMPTS:
        prompt_ids = enc.encode(pfn(name))
        gen = greedy_continue(model, prompt_ids, device, max_new=20)
        prompt_str = pfn(name).strip()
        print(f"  [{attr:12s}]  '{prompt_str}' → {gen[:60]}")
    print("-" * 60)

# ── TEST 2: actual 512-token training context ─────────────────────────────────

print("\n" + "=" * 60)
print("TEST 2: ACTUAL TRAINING CONTEXT (512 tokens ending after name)")
print("  ACTUAL = what's really in training data")
print("  GEN    = what the model generates")
print("=" * 60)

for name, gold in NAMES:
    pos, prefix = find_name_in_tokens(name, tokens, enc)
    if pos == -1:
        print(f"\n[{name}] NOT FOUND in token array!")
        continue

    name_ids = enc.encode(prefix)
    name_end  = pos + len(name_ids)          # position right after the name
    ctx_start = max(0, name_end - 512)
    context   = list(tokens[ctx_start:name_end])  # up to 512 tokens, ending after name

    x = torch.tensor([context], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(80):
            logits, _ = model(x[:, -512:])
            next_tok = logits[0, -1, :].argmax().item()
            if next_tok == EOT:
                break
            x = torch.cat([x, torch.tensor([[next_tok]], device=device)], dim=1)

    gen_continuation    = enc.decode(x[0, len(context):].tolist()).strip()
    actual_continuation = enc.decode(list(tokens[name_end:name_end + 80])).strip()

    print(f"\nNAME: {name}  (token pos {pos}, context len {len(context)})")
    print(f"GOLD: {gold}")
    print(f"ACTUAL: {actual_continuation[:200]}")
    print(f"GEN:    {gen_continuation[:200]}")
    print("-" * 60)
