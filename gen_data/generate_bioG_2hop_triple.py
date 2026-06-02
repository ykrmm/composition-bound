"""
Format: bioG_2hop_triple — structured triples with 2-hop information.

Two modes controlled by --explicit flag:

  IMPLICIT (default):
    - 8 focal triples:  [ENTITY] X [RELATION] attr        [VALUE] val
    - 3–5 friend scalar triples: [ENTITY] X [RELATION] friend.attr [VALUE] val
    - 3–5 enemy  scalar triples: [ENTITY] X [RELATION] enemy.attr  [VALUE] val
    - 2 friend relational 2-hop: friend.friend, friend.enemy
    - 2 enemy  relational 2-hop: enemy.friend, enemy.enemy
    Bridge Y is hidden — only appears in focal triple [ENTITY] X [RELATION] friend [VALUE] Y.

  EXPLICIT (--explicit):
    - 8 focal triples:  [ENTITY] X [RELATION] attr        [VALUE] val
    - 3–5 friend scalar triples: [ENTITY] Y [RELATION] attr [VALUE] val  (Y as subject)
    - 3–5 enemy  scalar triples: [ENTITY] W [RELATION] attr [VALUE] val  (W as subject)
    - 2 friend relational 2-hop: [ENTITY] Y [RELATION] friend/enemy [VALUE] Z
    - 2 enemy  relational 2-hop: [ENTITY] W [RELATION] friend/enemy [VALUE] Z
    Bridge Y/W appears explicitly as subject of the 2nd-hop triple.

All lines are permuted together as a single block (deterministic, seed =
person_id * 1000 + rep), so the model sees no fixed positional ordering.

IMPORTANT — train/test split:
  Only individuals with id < max_individuals (default 50000) get 2-hop triples.
  Individuals 50000..99999 are the QA test split (qa_{k}hop_val.jsonl) and must
  NOT see their 2-hop facts during pretrain — otherwise P_test would measure
  retrieval, not composition.

Outputs (implicit):
  graph_bios_data/bioG_2hop_triple_50k_sample.txt
  graph_bios_data/bioG_2hop_triple_50k_all.txt

Outputs (explicit):
  graph_bios_data/bioG_2hop_triple_explicit_50k_sample.txt
  graph_bios_data/bioG_2hop_triple_explicit_50k_all.txt

Usage:
    python generate_bioG_2hop_triple.py
    python generate_bioG_2hop_triple.py --explicit
    python generate_bioG_2hop_triple.py --sample_only
"""
from __future__ import annotations

import argparse
from typing import List

from tqdm import tqdm

from bioG_common import (ATTRS_8, attribute_value, fullname,
                          load_individuals, permute_indices,
                          preview_first_n, select_neighbor_attributes,
                          write_dataset)
from tokenizer_graph_bios import TOK_ENTITY, TOK_RELATION, TOK_VALUE


# ---------------------------------------------------------------------------

def _focal_value(person: dict, attr: str, individuals: List[dict]) -> str:
    if attr == "friend":
        return fullname(individuals[person["friend_id"]])
    if attr == "enemy":
        return fullname(individuals[person["enemy_id"]])
    return attribute_value(person, attr)


def build_block(person: dict, rep: int, individuals: List[dict],
                k_min: int = 3, k_max: int = 5, explicit: bool = False) -> List[str]:
    """Return all lines for one (person, rep) block, globally permuted."""
    name = fullname(person)

    # 8 focal triples (same in both modes)
    lines = []
    for attr in ATTRS_8:
        val = _focal_value(person, attr, individuals)
        lines.append(f"{TOK_ENTITY} {name} {TOK_RELATION} {attr} {TOK_VALUE} {val}")

    friend = individuals[person["friend_id"]]
    friend_name = fullname(friend)
    friend_attrs = select_neighbor_attributes(person["id"], rep, k=_k(person["id"], rep, k_min, k_max, salt=1), salt=1)

    if explicit:
        # [ENTITY] Y [RELATION] attr [VALUE] val  — Y is explicit subject (e1,r1,e2,r2,e3)
        for attr in friend_attrs:
            val = attribute_value(friend, attr)
            lines.append(f"{TOK_ENTITY} {friend_name} {TOK_RELATION} {attr} {TOK_VALUE} {val}")
        lines.append(f"{TOK_ENTITY} {friend_name} {TOK_RELATION} friend {TOK_VALUE} {fullname(individuals[friend['friend_id']])}")
        lines.append(f"{TOK_ENTITY} {friend_name} {TOK_RELATION} enemy  {TOK_VALUE} {fullname(individuals[friend['enemy_id']])}")
    else:
        # [ENTITY] X [RELATION] friend.attr [VALUE] val  — Y hidden, composed notation (e1,r1,r2,e3)
        for attr in friend_attrs:
            val = attribute_value(friend, attr)
            lines.append(f"{TOK_ENTITY} {name} {TOK_RELATION} friend.{attr} {TOK_VALUE} {val}")
        lines.append(f"{TOK_ENTITY} {name} {TOK_RELATION} friend.friend {TOK_VALUE} {fullname(individuals[friend['friend_id']])}")
        lines.append(f"{TOK_ENTITY} {name} {TOK_RELATION} friend.enemy  {TOK_VALUE} {fullname(individuals[friend['enemy_id']])}")

    enemy = individuals[person["enemy_id"]]
    enemy_name = fullname(enemy)
    enemy_attrs = select_neighbor_attributes(person["id"], rep, k=_k(person["id"], rep, k_min, k_max, salt=2), salt=2)

    if explicit:
        for attr in enemy_attrs:
            val = attribute_value(enemy, attr)
            lines.append(f"{TOK_ENTITY} {enemy_name} {TOK_RELATION} {attr} {TOK_VALUE} {val}")
        lines.append(f"{TOK_ENTITY} {enemy_name} {TOK_RELATION} friend {TOK_VALUE} {fullname(individuals[enemy['friend_id']])}")
        lines.append(f"{TOK_ENTITY} {enemy_name} {TOK_RELATION} enemy  {TOK_VALUE} {fullname(individuals[enemy['enemy_id']])}")
    else:
        for attr in enemy_attrs:
            val = attribute_value(enemy, attr)
            lines.append(f"{TOK_ENTITY} {name} {TOK_RELATION} enemy.{attr} {TOK_VALUE} {val}")
        lines.append(f"{TOK_ENTITY} {name} {TOK_RELATION} enemy.friend {TOK_VALUE} {fullname(individuals[enemy['friend_id']])}")
        lines.append(f"{TOK_ENTITY} {name} {TOK_RELATION} enemy.enemy  {TOK_VALUE} {fullname(individuals[enemy['enemy_id']])}")

    # Permute ALL lines together
    perm = permute_indices(len(lines), person["id"], rep)
    return [lines[i] for i in perm]


def _k(person_id: int, rep: int, k_min: int, k_max: int, salt: int) -> int:
    """Deterministic k in [k_min, k_max] for (person_id, rep, salt)."""
    import random
    return random.Random(person_id * 1000 + rep + salt * 99991).randint(k_min, k_max)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--individuals",    default="graph_bios_data/individuals.json")
    ap.add_argument("--out_full",       default=None)
    ap.add_argument("--out_sample",     default=None)
    ap.add_argument("--explicit",       action="store_true",
                    help="Explicit mode: use [ENTITY] Y [RELATION] attr [VALUE] val "
                         "instead of composed friend.attr notation (Ye et al. e1,r1,e2,r2,e3).")
    ap.add_argument("--sample_only",    action="store_true")
    ap.add_argument("--n_reps",         type=int, default=5)
    ap.add_argument("--shuffle_seed",   type=int, default=12345)
    ap.add_argument("--max_individuals", type=int, default=50000,
                    help="Only generate 2-hop for individuals 0..max_individuals-1 "
                         "(ids 0..49999 = QA train split). Prevents P_test data "
                         "leakage into pretrain.")
    args = ap.parse_args()

    variant = "explicit" if args.explicit else "implicit"
    out_full   = args.out_full   or f"graph_bios_data/bioG_2hop_triple_{variant}_50k_all.txt"
    out_sample = args.out_sample or f"graph_bios_data/bioG_2hop_triple_{variant}_50k_sample.txt"
    # keep original filenames for implicit (backward compat)
    if not args.explicit:
        out_full   = args.out_full   or "graph_bios_data/bioG_2hop_triple_50k_all.txt"
        out_sample = args.out_sample or "graph_bios_data/bioG_2hop_triple_50k_sample.txt"

    individuals = load_individuals(args.individuals)
    train_individuals = individuals[:args.max_individuals]
    print(f"[bioG_2hop_triple/{variant}] using {len(train_individuals):,} / {len(individuals):,} individuals "
          f"(ids 0..{args.max_individuals - 1} = QA train split)")

    # QC sample (first 50 train individuals × 2 reps)
    sample_records = []
    for pid in range(min(50, len(train_individuals))):
        for rep in range(2):
            sample_records.append(build_block(train_individuals[pid], rep, individuals, explicit=args.explicit))
    write_dataset(sample_records, out_sample, shuffle_seed=999)
    preview_first_n(out_sample, n_blocks=3)

    if args.sample_only:
        return

    # Full dataset — train individuals only
    print(f"\n[bioG_2hop_triple/{variant}] generating {len(train_individuals):,} × {args.n_reps} blocks...")
    records: List[List[str]] = []
    for person in tqdm(train_individuals, desc=f"building 2hop triples ({variant})"):
        for rep in range(args.n_reps):
            records.append(build_block(person, rep, individuals, explicit=args.explicit))
    write_dataset(records, out_full, shuffle_seed=args.shuffle_seed)
    preview_first_n(out_full, n_blocks=3)


if __name__ == "__main__":
    main()
