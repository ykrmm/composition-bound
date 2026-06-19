from __future__ import annotations

import json
import os
import random
from typing import List

EOS = "<|endoftext|>"

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
    rng = random.Random(person_id * 1000 + rep + salt * 13337)
    return rng.sample(ATTRS_6, k)


def permute_indices(n: int, person_id: int, rep: int) -> List[int]:
    indices = list(range(n))
    random.Random(person_id * 1000 + rep).shuffle(indices)
    return indices


def write_dataset(records: List[List[str]], path: str, shuffle_seed: int = 12345) -> None:
    flat = ["\n".join(lines) for lines in records]
    rng = random.Random(shuffle_seed)
    rng.shuffle(flat)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for text in flat:
            f.write(text.strip())
            f.write(f"\n{EOS}\n")
    print(f"  wrote {len(flat):,} entries -> {path}")


def estimate_tokens_per_individual(
    build_fn,
    individuals: List[dict],
    n_reps: int = 1,
    sample_size: int = 200,
) -> float:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    count = min(sample_size, len(individuals))
    total = 0
    for pid in range(count):
        for rep in range(n_reps):
            lines = build_fn(pid, rep)
            total += len(enc.encode("\n".join(lines)))
    return total / count / n_reps


def compute_target_tokens_default(ref_npy_path: str) -> int:
    import numpy as np
    return int(np.load(ref_npy_path).shape[0])


def preview_first_n(path: str, n_blocks: int = 3) -> None:
    with open(path) as f:
        raw = f.read()
    blocks = [b.strip() for b in raw.split(EOS) if b.strip()]
    print(f"\n--- {path} (first {n_blocks}) ---")
    for b in blocks[:n_blocks]:
        print(b)
        print()
