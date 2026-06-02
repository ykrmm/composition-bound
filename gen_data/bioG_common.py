"""
Shared utilities for bioG pretrain data generation.

Used by generate_bioG_2hop_nl.py, generate_bioG_2hop_triple.py, and mix_pretrain.py.
"""
from __future__ import annotations

import json
import os
import random
from typing import List

EOS = "<|endoftext|>"

# 6 scalar attributes used in 2-hop augmentation
ATTRS_6 = ["birth_date", "birth_city", "university", "major", "company", "company_city"]
ATTRS_8 = ATTRS_6 + ["friend", "enemy"]


def fullname(person: dict) -> str:
    return f"{person['first_name']} {person['middle_name']} {person['last_name']}"


def load_individuals(path: str) -> List[dict]:
    with open(path) as f:
        return json.load(f)


def attribute_value(person: dict, attr: str) -> str:
    if attr == "birth_date":
        return f"{person['birthmonth']} {person['birthday']}, {person['birthyear']}"
    if attr == "birth_city":
        return person["birthcity"]
    if attr == "university":
        return person["university"]
    if attr == "major":
        return person["field"]
    if attr == "company":
        return person["company1name"]
    if attr == "company_city":
        return person["company1city"]
    raise KeyError(f"unknown attribute: {attr!r}")


def select_neighbor_attributes(person_id: int, rep: int, k: int, salt: int) -> List[str]:
    """Return k distinct scalar attributes, deterministic for (person_id, rep, salt)."""
    rng = random.Random(person_id * 1000 + rep + salt * 13337)
    return rng.sample(ATTRS_6, k)


def permute_indices(n: int, person_id: int, rep: int) -> List[int]:
    """Return a deterministic permutation of range(n) for (person_id, rep)."""
    indices = list(range(n))
    random.Random(person_id * 1000 + rep).shuffle(indices)
    return indices


def write_dataset(records: List[List[str]], path: str, shuffle_seed: int = 12345) -> None:
    """Write a list of records (each a list of lines) to path, shuffled, EOS-separated."""
    flat = ["\n".join(lines) for lines in records]
    rng = random.Random(shuffle_seed)
    rng.shuffle(flat)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for text in flat:
            f.write(text.strip())
            f.write(f"\n{EOS}\n")
    print(f"  wrote {len(flat):,} entries → {path}")


def preview_first_n(path: str, n_blocks: int = 3) -> None:
    """Print the first n_blocks from an EOS-separated file."""
    with open(path) as f:
        raw = f.read()
    blocks = [b.strip() for b in raw.split(EOS) if b.strip()]
    print(f"\n--- {path} (first {n_blocks} blocks) ---")
    for b in blocks[:n_blocks]:
        print(b)
        print()
