"""
Mix N pretrain corpora into a single shuffled file with arbitrary ratios.

Mix any combination of:
- bioS_multi5p_fullname (NL biographies)
- bioG_triple, bioG_neighbor, bioG_2hop_explicit, bioG_2hop_triple (graph formats)
- any other <|endoftext|>-separated corpus

The mix is at the BLOCK level: each EOS-separated entry is one sample. We:
1. Read each input file, split on <|endoftext|>.
2. For each source, sub-sample WITHOUT replacement to hit the target ratio.
3. Concatenate all retained blocks and shuffle globally.

The total cap is the binding constraint: we never duplicate blocks, so the
total size = min_i(n_blocks_i / ratio_i). Increase a source's available blocks
(more reps in its generator) if you want a larger mixed corpus.

Usage examples:
    # 50/50 NL + triples
    python mix_pretrain.py \
        --sources graph_bios_data/bios_multi5p_fullname_all.txt \
                  graph_bios_data/bioG_triple_all.txt \
        --ratios  0.5 0.5 \
        --out     graph_bios_data/mixed_triple_50_50_all.txt

    # 3-way: 30 % NL + 35 % triples + 35 % neighbor
    python mix_pretrain.py \
        --sources graph_bios_data/bios_multi5p_fullname_all.txt \
                  graph_bios_data/bioG_triple_all.txt \
                  graph_bios_data/bioG_neighbor_all.txt \
        --ratios  0.30 0.35 0.35 \
        --out     graph_bios_data/mixed_3way_all.txt

    # Allen-Zhu 0.8/0.2 (graph dominant), one source on each side
    python mix_pretrain.py \
        --sources graph_bios_data/bios_multi5p_fullname_all.txt \
                  graph_bios_data/bioG_2hop_explicit_all.txt \
        --ratios  0.2 0.8 \
        --out     graph_bios_data/mixed_2hop_explicit_20_80_all.txt
"""
from __future__ import annotations

import argparse
import os
import random
from typing import List

from tokenizer_graph_bios import enc_bioG, TOK_EOS


def _read_blocks(path: str) -> List[str]:
    with open(path) as f:
        text = f.read()
    return [b.strip() for b in text.split(TOK_EOS) if b.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", required=True, nargs="+",
                    help="One or more pretrain .txt files (EOS-separated)")
    ap.add_argument("--ratios", required=True, nargs="+", type=float,
                    help="Target fraction for each source (will be normalised "
                         "to sum to 1)")
    ap.add_argument("--out", required=True, help="Output mixed .txt")
    ap.add_argument("--shuffle_seed", type=int, default=0)
    ap.add_argument("--max_total_blocks", type=int, default=0,
                    help="Cap on total blocks (0 = no cap; binding constraint = "
                         "smallest source / its ratio).")
    args = ap.parse_args()

    if len(args.sources) != len(args.ratios):
        raise SystemExit(
            f"--sources ({len(args.sources)}) and --ratios "
            f"({len(args.ratios)}) must have equal length")

    rng = random.Random(args.shuffle_seed)

    # ---- read & normalise ------------------------------------------------
    src_blocks = []
    for path in args.sources:
        blocks = _read_blocks(path)
        rng.shuffle(blocks)   # shuffle within source so subsampling is uniform
        src_blocks.append(blocks)
        print(f"[mix] {path:<60s}  {len(blocks):>10,} blocks")

    s = sum(args.ratios)
    if s <= 0:
        raise SystemExit("ratios must sum to a positive number")
    if not (0.99 <= s <= 1.01):
        print(f"[mix] WARNING: ratios sum to {s:.3f}, normalising to 1")
    ratios = [r / s for r in args.ratios]

    # ---- compute total cap (binding constraint) --------------------------
    cap = min(len(blocks) / r for blocks, r in zip(src_blocks, ratios) if r > 0)
    if args.max_total_blocks > 0:
        cap = min(cap, args.max_total_blocks)
    cap = int(cap)
    keep = [int(round(r * cap)) for r in ratios]
    # adjust last to make sums exact
    keep[-1] = cap - sum(keep[:-1])
    print(f"[mix] target ratios   : {[f'{r:.2f}' for r in ratios]}")
    print(f"[mix] kept blocks     : {keep}  (total {sum(keep):,})")
    achieved = [k / cap for k in keep]
    print(f"[mix] achieved ratios : {[f'{r:.3f}' for r in achieved]}")

    # ---- subsample, concatenate, shuffle ---------------------------------
    combined: List[str] = []
    for path, blocks, k in zip(args.sources, src_blocks, keep):
        combined.extend(blocks[:k])
    rng.shuffle(combined)

    # ---- write -----------------------------------------------------------
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"[mix] writing {len(combined):,} blocks → {args.out}")
    with open(args.out, "w") as f:
        for b in combined:
            f.write(b)
            f.write(f"\n{TOK_EOS}\n")

    # ---- token count summary ---------------------------------------------
    print("[mix] estimating total tokens (1 % sample)...")
    sample = combined[: max(1, len(combined) // 100)]
    sample_text = ("\n" + TOK_EOS + "\n").join(sample) + "\n"
    sample_tokens = len(enc_bioG.encode(sample_text, allowed_special="all"))
    est_total = int(sample_tokens * len(combined) / max(1, len(sample)))
    print(f"[mix] estimated total tokens ≈ {est_total:,}")
    print("[mix] done.")


if __name__ == "__main__":
    main()
