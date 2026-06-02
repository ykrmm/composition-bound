"""
Format: bioG_2hop_nl — pure natural language bio (multi5p + permute) with
explicit 2-hop sentences.

Each individual gets n_reps (default 5) permuted repetitions, matching the
bios_multi5p_fullname scheme from generate_graph_bios.py:
  - Each rep picks different random templates (from the ~50 per attribute pools)
  - Each rep has a different sentence permutation via augmentation_permutation2
  - Seed = person_id * 1000 + rep for full reproducibility

Structure of one (person, rep) block:
  1. 6 NL sentences about X + 1-hop relation sentences (X's friend is Y /
     X's enemy is Z), all shuffled together via augmentation_permutation2
     (fullname mode, ~8 sentences total).
  2. 3–5 NL 2-hop sentences about Y (X's friend), pattern:
         "{X}'s friend {Y} [attribute phrase]"
  3. 3–5 NL 2-hop sentences about Z (X's enemy), pattern:
         "{X}'s enemy {Z} [attribute phrase]"
  Blocks 2 & 3 appear in a random order that varies per rep (deterministic).

Estimated block size: ~230–300 tokens → ~2 blocks per 512-token context window.

Outputs:
  graph_bios_data/bioG_2hop_nl_sample.txt  (50 individuals × 2 reps — for QC)
  graph_bios_data/bioG_2hop_nl_all.txt     (full 100K × n_reps — for pretrain)

Usage:
    python generate_bioG_2hop_nl.py            # full + sample
    python generate_bioG_2hop_nl.py --sample_only
"""
from __future__ import annotations

import argparse
import os
import random
from typing import List

from tqdm import tqdm

from bioG_common import (
    ATTRS_6,
    fullname as _fullname,
    load_individuals,
    select_neighbor_attributes,
)
from generate_graph_bios import (
    get_text_simple3,
    augmentation_permutation2,
    append_graph_sentences,
)
from tokenizer_graph_bios import enc_bioG, TOK_EOS


# ---------------------------------------------------------------------------
# 2-hop sentence templates
# ---------------------------------------------------------------------------
# Explicit (default): "{x}'s {rel} {y} [phrase]" — bridge Y named in 2-hop sentence
# Implicit (--implicit): "{x}'s {rel} [phrase]"  — bridge Y unnamed in 2-hop sentence
# Y is always present in X's bio via 1-hop relation sentences regardless of mode.

TWOHOP_TEMPLATES = {
    "birth_date": [
        "{x}'s {rel} {y} was born on {birthday}.",
        "{x}'s {rel} {y}'s birth date is {birthday}.",
        "The birth date of {x}'s {rel} {y} is {birthday}.",
        "{x}'s {rel} {y}'s birthday falls on {birthday}.",
    ],
    "birth_city": [
        "{x}'s {rel} {y} was born in {birthcity}.",
        "{x}'s {rel} {y}'s birth city is {birthcity}.",
        "The birth city of {x}'s {rel} {y} is {birthcity}.",
        "{x}'s {rel} {y} hails from {birthcity}.",
    ],
    "university": [
        "{x}'s {rel} {y} studied at {university}.",
        "{x}'s {rel} {y}'s university is {university}.",
        "The university of {x}'s {rel} {y} is {university}.",
        "{x}'s {rel} {y} graduated from {university}.",
    ],
    "major": [
        "{x}'s {rel} {y} majored in {field}.",
        "{x}'s {rel} {y}'s major is {field}.",
        "The major of {x}'s {rel} {y} is {field}.",
        "{x}'s {rel} {y} specialized in {field}.",
    ],
    "company": [
        "{x}'s {rel} {y} worked at {company1name}.",
        "{x}'s {rel} {y}'s employer is {company1name}.",
        "The employer of {x}'s {rel} {y} is {company1name}.",
        "{x}'s {rel} {y} was employed by {company1name}.",
    ],
    "company_city": [
        "{x}'s {rel} {y} worked in {company1city}.",
        "{x}'s {rel} {y}'s work city is {company1city}.",
        "The work city of {x}'s {rel} {y} is {company1city}.",
        "{x}'s {rel} {y} developed their career in {company1city}.",
    ],
    "friend": [
        "{x}'s {rel} {y}'s friend is {friend_name}.",
        "The friend of {x}'s {rel} {y} is {friend_name}.",
    ],
    "enemy": [
        "{x}'s {rel} {y}'s enemy is {enemy_name}.",
        "The enemy of {x}'s {rel} {y} is {enemy_name}.",
    ],
}

IMPLICIT_TWOHOP_TEMPLATES = {
    "birth_date": [
        "{x}'s {rel} was born on {birthday}.",
        "{x}'s {rel}'s birth date is {birthday}.",
        "The birth date of {x}'s {rel} is {birthday}.",
        "{x}'s {rel}'s birthday falls on {birthday}.",
    ],
    "birth_city": [
        "{x}'s {rel} was born in {birthcity}.",
        "{x}'s {rel}'s birth city is {birthcity}.",
        "The birth city of {x}'s {rel} is {birthcity}.",
        "{x}'s {rel} hails from {birthcity}.",
    ],
    "university": [
        "{x}'s {rel} studied at {university}.",
        "{x}'s {rel}'s university is {university}.",
        "The university of {x}'s {rel} is {university}.",
        "{x}'s {rel} graduated from {university}.",
    ],
    "major": [
        "{x}'s {rel} majored in {field}.",
        "{x}'s {rel}'s major is {field}.",
        "The major of {x}'s {rel} is {field}.",
        "{x}'s {rel} specialized in {field}.",
    ],
    "company": [
        "{x}'s {rel} worked at {company1name}.",
        "{x}'s {rel}'s employer is {company1name}.",
        "The employer of {x}'s {rel} is {company1name}.",
        "{x}'s {rel} was employed by {company1name}.",
    ],
    "company_city": [
        "{x}'s {rel} worked in {company1city}.",
        "{x}'s {rel}'s work city is {company1city}.",
        "The work city of {x}'s {rel} is {company1city}.",
        "{x}'s {rel} developed their career in {company1city}.",
    ],
    "friend": [
        "{x}'s {rel}'s friend is {friend_name}.",
        "The friend of {x}'s {rel} is {friend_name}.",
    ],
    "enemy": [
        "{x}'s {rel}'s enemy is {enemy_name}.",
        "The enemy of {x}'s {rel} is {enemy_name}.",
    ],
}


def _attr_kwargs(neighbor: dict, attr: str, individuals: list | None = None) -> dict:
    """Return format kwargs for a 2-hop template given a neighbor and attribute."""
    if attr == "birth_date":
        return {"birthday": f"{neighbor['birthmonth']} {neighbor['birthday']}, {neighbor['birthyear']}"}
    if attr == "birth_city":
        return {"birthcity": neighbor["birthcity"]}
    if attr == "university":
        return {"university": neighbor["university"]}
    if attr == "major":
        return {"field": neighbor["field"]}
    if attr == "company":
        return {"company1name": neighbor["company1name"]}
    if attr == "company_city":
        return {"company1city": neighbor["company1city"]}
    if attr == "friend" and individuals is not None:
        return {"friend_name": _fullname(individuals[neighbor["friend_id"]])}
    if attr == "enemy" and individuals is not None:
        return {"enemy_name": _fullname(individuals[neighbor["enemy_id"]])}
    raise KeyError(attr)


def _build_2hop_sentences(x_name: str, rel_word: str, neighbor: dict,
                           attrs: List[str], person_id: int, rep: int,
                           salt: int, individuals: list | None = None,
                           implicit: bool = False) -> List[str]:
    """One sentence per attr, template chosen deterministically via local RNG."""
    y_name = _fullname(neighbor)
    rng = random.Random(person_id * 1000 + rep + salt * 13337)
    templates = IMPLICIT_TWOHOP_TEMPLATES if implicit else TWOHOP_TEMPLATES
    sentences = []
    for attr in attrs:
        tmpl = rng.choice(templates[attr])
        kwargs = _attr_kwargs(neighbor, attr, individuals)
        if not implicit:
            kwargs["y"] = y_name
        sentences.append(tmpl.format(x=x_name, rel=rel_word, **kwargs))
    return sentences


# ---------------------------------------------------------------------------
# Block builder
# ---------------------------------------------------------------------------

def build_block(person: dict, rep: int, individuals: List[dict],
                k_min: int = 3, k_max: int = 5,
                implicit: bool = False) -> str:
    """Build one (person, rep) text block as a single string."""
    random.seed(person["id"] * 1000 + rep)

    # 1. Focal bio (6 NL sentences) + 1-hop friend/enemy, shuffled together
    bio = get_text_simple3(person, fullname=True)
    bio = append_graph_sentences(bio, person, individuals)
    bio = augmentation_permutation2(person, bio, fullname=True).strip()

    # 2. Attribute counts for 2-hop (3–5 per neighbor, varies per rep)
    rng_k = random.Random(person["id"] * 1000 + rep + 99991)
    k_friend = rng_k.randint(k_min, k_max)
    k_enemy  = rng_k.randint(k_min, k_max)
    friend_attrs = select_neighbor_attributes(person["id"], rep, k=k_friend, salt=1)
    enemy_attrs  = select_neighbor_attributes(person["id"], rep, k=k_enemy,  salt=2)

    # 3. 2-hop sentences for friend and enemy (scalar + relational)
    x_name  = _fullname(person)
    friend  = individuals[person["friend_id"]]
    enemy   = individuals[person["enemy_id"]]
    friend_attrs_full = list(friend_attrs) + ["friend", "enemy"]
    enemy_attrs_full  = list(enemy_attrs)  + ["friend", "enemy"]
    friend_sents = _build_2hop_sentences(x_name, "friend", friend,
                                          friend_attrs_full, person["id"], rep,
                                          salt=1, individuals=individuals,
                                          implicit=implicit)
    enemy_sents  = _build_2hop_sentences(x_name, "enemy",  enemy,
                                          enemy_attrs_full,  person["id"], rep,
                                          salt=2, individuals=individuals,
                                          implicit=implicit)

    # 4. Random order of friend/enemy blocks (deterministic per rep)
    rng_order = random.Random(person["id"] * 1000 + rep + 77771)
    blocks = [friend_sents, enemy_sents]
    rng_order.shuffle(blocks)

    twohop = " ".join(blocks[0] + blocks[1])
    return bio + " " + twohop


def build_block_no2hop(person: dict, rep: int, individuals: List[dict]) -> str:
    """Bio-only block for P_test individuals (no 2-hop sentences).

    Same 1-hop bio as the first part of build_block(), ensuring P_test individuals
    appear in pretrain so the model knows who they are — but without any 2-hop
    facts, so 2-hop eval measures composition, not recall.
    """
    random.seed(person["id"] * 1000 + rep)
    bio = get_text_simple3(person, fullname=True)
    bio = append_graph_sentences(bio, person, individuals)
    bio = augmentation_permutation2(person, bio, fullname=True).strip()
    return bio


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _estimate_tokens(train_individuals: List[dict], all_individuals: List[dict],
                     n_sample: int = 200, implicit: bool = False) -> float:
    """Average tokens per block (including EOS) on a random sample."""
    rng = random.Random(7)
    sample = rng.sample(train_individuals, min(n_sample, len(train_individuals)))
    total = 0
    for person in sample:
        text = build_block(person, rep=0, individuals=all_individuals, implicit=implicit)
        total += len(enc_bioG.encode(text, allowed_special="all")) + 1
    return total / len(sample)


def write_txt(texts: List[str], out_path: str, shuffle_seed: int = 12345) -> None:
    rng = random.Random(shuffle_seed)
    rng.shuffle(texts)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for text in texts:
            f.write(text.strip())
            f.write(f"\n{TOK_EOS}\n")
    print(f"[bioG_2hop_nl] wrote {len(texts):,} blocks → {out_path}")


def preview(out_path: str, n: int = 3) -> None:
    with open(out_path) as f:
        raw = f.read()
    blocks = [b.strip() for b in raw.split(TOK_EOS) if b.strip()]
    print(f"\n--- {out_path} (first {n} blocks) ---")
    for b in blocks[:n]:
        ntok = len(enc_bioG.encode(b, allowed_special="all"))
        print(b)
        print(f"  [{ntok} tokens]\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--individuals",    default="graph_bios_data/individuals.json")
    ap.add_argument("--out_full",       default=None,
                    help="Output path for full dataset (auto-named if not set).")
    ap.add_argument("--out_sample",     default=None,
                    help="Output path for QC sample (auto-named if not set).")
    ap.add_argument("--sample_only",    action="store_true")
    ap.add_argument("--implicit",       action="store_true",
                    help="Use implicit templates: bridge Y is NOT named in 2-hop "
                         "sentences (Y is still present via 1-hop bio sentences).")
    ap.add_argument("--n_reps",         type=int, default=5)
    ap.add_argument("--shuffle_seed",   type=int, default=12345)
    ap.add_argument("--n_sample",       type=int, default=50)
    ap.add_argument("--max_individuals", type=int, default=50000,
                    help="Only generate 2-hop for ids 0..max_individuals-1 (P_train).")
    args = ap.parse_args()

    variant = "implicit" if args.implicit else "explicit"
    if args.out_full is None:
        args.out_full   = f"graph_bios_data/bioG_2hop_nl_{variant}_50k_all.txt"
    if args.out_sample is None:
        args.out_sample = f"graph_bios_data/bioG_2hop_nl_{variant}_50k_sample.txt"

    individuals = load_individuals(args.individuals)
    train_individuals = individuals[:args.max_individuals]
    print(f"[bioG_2hop_nl_{variant}] {len(train_individuals):,} P_train / "
          f"{len(individuals):,} total  (ids 0..{args.max_individuals - 1})")

    # QC sample
    print(f"\nBuilding QC sample ({args.n_sample} individuals × 2 reps, {variant})...")
    sample_texts = []
    for person in tqdm(train_individuals[:args.n_sample], desc="sample"):
        for rep in range(2):
            sample_texts.append(build_block(person, rep, individuals, implicit=args.implicit))
    write_txt(sample_texts, args.out_sample, shuffle_seed=args.shuffle_seed)
    preview(args.out_sample)

    if args.sample_only:
        return

    # Full dataset — P_train with 2-hop, P_test with bio only
    test_individuals = individuals[args.max_individuals:]
    print(f"\nEstimating tokens per block ({variant})...")
    tok_per_block = _estimate_tokens(train_individuals, individuals, implicit=args.implicit)
    print(f"  ~{tok_per_block:.0f} tokens/block (EOS included)")

    print(f"\nBuilding full dataset ({variant})...")
    print(f"  P_train ({len(train_individuals):,} × {args.n_reps} reps): bio + 2-hop sentences")
    print(f"  P_test  ({len(test_individuals):,} × {args.n_reps} reps): bio only (no 2-hop)")
    all_texts = []
    for person in tqdm(train_individuals, desc=f"P_train (bio+2hop {variant})"):
        for rep in range(args.n_reps):
            all_texts.append(build_block(person, rep, individuals, implicit=args.implicit))
    for person in tqdm(test_individuals, desc="P_test  (bio only)"):
        for rep in range(args.n_reps):
            all_texts.append(build_block_no2hop(person, rep, individuals))
    write_txt(all_texts, args.out_full, shuffle_seed=args.shuffle_seed)


if __name__ == "__main__":
    main()
