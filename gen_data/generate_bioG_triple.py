"""
Generates bioG_triple: 1-hop atomic structured triples for all individuals.

Each block contains 8 [ENTITY]/[RELATION]/[VALUE] lines (one per attribute),
in a deterministic per-individual permutation across reps.

Outputs:
    bioG_triple_sample.txt  -- 50 individuals x 2 reps (sanity check)
    bioG_triple_all.txt     -- full dataset for pretraining

Usage:
    python generate_bioG_triple.py
    python generate_bioG_triple.py --sample_only
"""
from __future__ import annotations

import argparse
from typing import List

from tqdm import tqdm

from bioG_common import (ATTRS_8, attribute_value, compute_target_tokens_default,
                          estimate_tokens_per_individual, fullname, load_individuals,
                          permute_indices, preview_first_n, write_dataset)
from tokenizer_graph_bios import TOK_ENTITY, TOK_RELATION, TOK_VALUE


def _value_for_attr(person: dict, attr: str, individuals: List[dict]) -> str:
    if attr == "friend":
        return fullname(individuals[person["friend_id"]])
    if attr == "enemy":
        return fullname(individuals[person["enemy_id"]])
    return attribute_value(person, attr)


def build_triples_one_rep(person: dict, rep: int, individuals: List[dict]) -> List[str]:
    name = fullname(person)
    base_lines = [
        f"{TOK_ENTITY} {name} {TOK_RELATION} {attr} {TOK_VALUE} {_value_for_attr(person, attr, individuals)}"
        for attr in ATTRS_8
    ]
    perm = permute_indices(len(base_lines), person["id"], rep)
    return [base_lines[i] for i in perm]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--individuals",  default="graph_bios_data/individuals.json")
    ap.add_argument("--out_full",     default="graph_bios_data/bioG_triple_all.txt")
    ap.add_argument("--out_sample",   default="graph_bios_data/bioG_triple_sample.txt")
    ap.add_argument("--sample_only",  action="store_true")
    ap.add_argument("--n_reps",       type=int, default=5)
    ap.add_argument("--shuffle_seed", type=int, default=12345)
    args = ap.parse_args()

    individuals = load_individuals(args.individuals)
    n_reps = args.n_reps

    tpi = estimate_tokens_per_individual(
        lambda pid, rep: build_triples_one_rep(individuals[pid], rep, individuals),
        individuals, n_reps=1, sample_size=200,
    )
    print(f"[bioG_triple] n_reps={n_reps}, ~{tpi:.0f} tokens/rep -> "
          f"~{int(n_reps * tpi * len(individuals)):,} total")

    sample_records = [
        build_triples_one_rep(individuals[pid], rep, individuals)
        for pid in range(50) for rep in range(2)
    ]
    write_dataset(sample_records, args.out_sample, shuffle_seed=999)
    preview_first_n(args.out_sample, n_blocks=4)

    if args.sample_only:
        return

    print(f"generating {len(individuals):,} x {n_reps} blocks...")
    records: List[List[str]] = []
    for person in tqdm(individuals):
        for rep in range(n_reps):
            records.append(build_triples_one_rep(person, rep, individuals))

    write_dataset(records, args.out_full, shuffle_seed=args.shuffle_seed)
    preview_first_n(args.out_full, n_blocks=3)


if __name__ == "__main__":
    main()
