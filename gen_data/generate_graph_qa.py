"""
Generate 1-hop / 2-hop / 3-hop QA pairs from graph_bios_data/individuals.json.

Split routing (same-split edges make this natural):
    qa_{k}hop_train.jsonl  -- questions about P_train individuals (ids 0..49999);
                              all traversed relations stay inside the train half.
    qa_{k}hop_val.jsonl    -- questions about P_test individuals (50000..99999);
                              all traversed relations stay inside the val half.

Hops (pairs per individual):
    1-hop: 8 attrs = 6 bio + friend + enemy                 -> 400K per split
    2-hop: 2 relations x 8 attrs                            -> 800K per split
    3-hop: 2 x 2 relations x 8 attrs                        -> 1.6M per split

Each record:
    {
      "id": 12345,
      "attr": "birthday",
      "chain": ["friend"],          # relation chain from subject to terminal
      "hops": 2,
      "prompt": "What is the birth date of X's friend? Answer:",
      "answer": "March 15, 1942",
      "text":   "What is the birth date of X's friend? Answer: March 15, 1942<|endoftext|>"
    }

Usage:
    python generate_graph_qa.py \
        --individuals graph_bios_data/individuals.json \
        --out_dir graph_qa_data
"""

import os
import json
import argparse


RELATIONS = [
    ("friend", "friend_id"),
    ("enemy",  "enemy_id"),
]


def _fullname(person):
    return f"{person['first_name']} {person['middle_name']} {person['last_name']}"


ATTRIBUTES = [
    {"key": "birthday",      "question": "What is the birth date of {subject}?",
     "answer": lambda p, ind: f"{p['birthmonth']} {p['birthday']}, {p['birthyear']}"},
    {"key": "birthcity",     "question": "What is the birth city of {subject}?",
     "answer": lambda p, ind: p["birthcity"]},
    {"key": "university",    "question": "Which university did {subject} study?",
     "answer": lambda p, ind: p["university"]},
    {"key": "major",         "question": "What major did {subject} study?",
     "answer": lambda p, ind: p["field"]},
    {"key": "employer",      "question": "Which company did {subject} work for?",
     "answer": lambda p, ind: p["company1name"]},
    {"key": "employer_city", "question": "Where did {subject} work?",
     "answer": lambda p, ind: p["company1city"]},
    {"key": "friend",        "question": "Who is the friend of {subject}?",
     "answer": lambda p, ind: _fullname(ind[p["friend_id"]])},
    {"key": "enemy",         "question": "Who is the enemy of {subject}?",
     "answer": lambda p, ind: _fullname(ind[p["enemy_id"]])},
]


def walk_relations(person, rel_keys, individuals):
    """Follow a chain of relation keys from person. Returns the terminal person."""
    p = person
    for rk in rel_keys:
        p = individuals[p[rk]]
    return p


def make_record(person, relation_chain, attr, individuals):
    """Build one QA record.
    relation_chain: list of (name, key) pairs, e.g. [("friend", "friend_id")].
    """
    subject = _fullname(person)
    for rel_name, _ in relation_chain:
        subject = f"{subject}'s {rel_name}"

    question = attr["question"].format(subject=subject)
    terminal = walk_relations(person, [rk for _, rk in relation_chain], individuals)
    answer   = attr["answer"](terminal, individuals)

    return {
        "id":    person["id"],
        "attr":  attr["key"],
        "chain": [r for r, _ in relation_chain],
        "hops":  len(relation_chain) + 1,
        "prompt": f"{question} Answer:",
        "answer": answer,
        "text":   f"{question} Answer: {answer}<|endoftext|>",
    }


def relation_chains_for_hops(hops):
    if hops == 1:
        return [[]]
    if hops == 2:
        return [[r] for r in RELATIONS]
    if hops == 3:
        return [[r1, r2] for r1 in RELATIONS for r2 in RELATIONS]
    raise ValueError(f"unsupported hops={hops}")


def write_split(people, individuals, hops, out_path):
    chains = relation_chains_for_hops(hops)
    count = 0
    with open(out_path, "w") as fout:
        for person in people:
            for chain in chains:
                for attr in ATTRIBUTES:
                    rec = make_record(person, chain, attr, individuals)
                    fout.write(json.dumps(rec) + "\n")
                    count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--individuals", default="../graph_bios_data/individuals.json")
    parser.add_argument("--out_dir",     default="../graph_qa_data")
    parser.add_argument("--hops", type=int, nargs="+", default=[1, 2, 3],
                        help="Hop counts to generate (default: 1 2 3)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.individuals) as f:
        individuals = json.load(f)

    # Sanity: every individual has edges, and edges point inside the same half.
    n = len(individuals)
    half = n // 2
    for p in individuals:
        assert "friend_id" in p and "enemy_id" in p, \
            f"individual {p['id']} missing friend_id/enemy_id"
        src_half = 0 if p["id"] < half else 1
        for rk in ("friend_id", "enemy_id"):
            tgt_half = 0 if p[rk] < half else 1
            assert src_half == tgt_half, \
                f"cross-split edge {p['id']} -> {p[rk]} via {rk}"

    splits = [("train", individuals[:half]), ("val", individuals[half:])]

    for hops in args.hops:
        for split, people in splits:
            out_path = os.path.join(args.out_dir, f"qa_{hops}hop_{split}.jsonl")
            count = write_split(people, individuals, hops, out_path)
            print(f"  [{hops}-hop {split}] {len(people):,} people -> "
                  f"{count:,} pairs -> {out_path}")


if __name__ == "__main__":
    main()
