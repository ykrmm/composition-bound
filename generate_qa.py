"""
Generate QA pairs from individuals.json for BioS QA evaluation.

6 attributes × N individuals → JSONL files with prompt / answer / full text.

Usage:
    python generate_qa.py --individuals bios_data/individuals.json --out_dir qa_data

Output:
    qa_data/qa_train.jsonl   -- 50K × 6 = 300K pairs  (P_train, IDs  0-49999)
    qa_data/qa_val.jsonl     -- 50K × 6 = 300K pairs  (P_test,  IDs 50000-99999)
"""

import os
import json
import argparse

# ── 6 QA templates (one per bio attribute) ────────────────────────────────────

ATTRIBUTES = [
    {
        "key":      "birthday",
        "question": "What is the birth date of {name}?",
        "answer":   lambda p: f"{p['birthmonth']} {p['birthday']}, {p['birthyear']}",
    },
    {
        "key":      "birthcity",
        "question": "What is the birth city of {name}?",
        "answer":   lambda p: p["birthcity"],
    },
    {
        "key":      "university",
        "question": "Which university did {name} study?",
        "answer":   lambda p: p["university"],
    },
    {
        "key":      "major",
        "question": "What major did {name} study?",
        "answer":   lambda p: p["field"],
    },
    {
        "key":      "employer",
        "question": "Which company did {name} work for?",
        "answer":   lambda p: p["company1name"],
    },
    {
        "key":      "employer_city",
        "question": "Where did {name} work?",
        "answer":   lambda p: p["company1city"],
    },
]


def make_record(person, attr):
    name     = f"{person['first_name']} {person['middle_name']} {person['last_name']}"
    question = attr["question"].format(name=name)
    answer   = attr["answer"](person)
    return {
        "id":     person["id"],
        "attr":   attr["key"],
        "prompt": f"{question} Answer:",          # fed to model during eval
        "answer": answer,                          # gold label for exact match
        "text":   f"{question} Answer: {answer}<|endoftext|>",  # full text for training
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--individuals", default="bios_data/individuals.json")
    parser.add_argument("--out_dir",     default="qa_data")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.individuals) as f:
        individuals = json.load(f)

    half   = len(individuals) // 2
    splits = [("train", individuals[:half]), ("val", individuals[half:])]

    for split, people in splits:
        out_path = os.path.join(args.out_dir, f"qa_{split}.jsonl")
        with open(out_path, "w") as fout:
            for person in people:
                for attr in ATTRIBUTES:
                    fout.write(json.dumps(make_record(person, attr)) + "\n")
        print(f"  [{split}] {len(people):,} individuals × {len(ATTRIBUTES)} attrs "
              f"= {len(people) * len(ATTRIBUTES):,} pairs → {out_path}")


if __name__ == "__main__":
    main()
