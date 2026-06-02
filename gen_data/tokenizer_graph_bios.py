"""
Augmented tiktoken encoding for the bioG family of pretrain formats
+ tokenization script (graph + text).

Adds 5 single-token markers on top of the GPT-2 BPE vocab:
- [ENTITY]      → 50257
- [RELATION]    → 50258
- [VALUE]       → 50259
- [LINKED]      → 50260
- [END_ENTITY]  → 50261

Vocab is padded to 50304 (multiple of 64 for tensor core alignment).

Usage:
    python tokenize_bioG.py --in_dir bioG_data --out_dir bioG_tokens

Input:
    .txt files with samples separated by <|endoftext|>
    containing graph tokens + natural language

Output:
    flat uint16 numpy arrays (.npy)
"""

import os
import argparse
import numpy as np
import tiktoken
from tqdm import tqdm


# =========================
# Tokenizer definition
# =========================

SPECIAL_TOKENS = {
    "<|endoftext|>": 50256,   # GPT-2 base
    "[ENTITY]":      50257,
    "[RELATION]":    50258,
    "[VALUE]":       50259,
    "[LINKED]":      50260,
    "[END_ENTITY]":  50261,
}

VOCAB_SIZE_BIOG = 50304

_base = tiktoken.get_encoding("gpt2")

enc_bioG = tiktoken.Encoding(
    name="gpt2_bioG",
    pat_str=_base._pat_str,
    mergeable_ranks=_base._mergeable_ranks,
    special_tokens=SPECIAL_TOKENS,
)

# Convenience tokens
TOK_ENTITY     = "[ENTITY]"
TOK_RELATION   = "[RELATION]"
TOK_VALUE      = "[VALUE]"
TOK_LINKED     = "[LINKED]"
TOK_END_ENTITY = "[END_ENTITY]"
TOK_EOS        = "<|endoftext|>"


# =========================
# Tokenization logic
# =========================

def tokenize_file(path: str) -> np.ndarray:
    eot = enc_bioG._special_tokens[TOK_EOS]

    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read()

    # Split samples on EOS
    samples = [s.strip() for s in raw.split(TOK_EOS) if s.strip()]
    print(f"  {len(samples):,} samples found")

    all_tokens = []

    for sample in tqdm(samples, desc="  tokenizing", unit="sample"):
        # IMPORTANT: preserve special tokens
        tokens = [eot] + enc_bioG.encode(sample, allowed_special="all")
        all_tokens.extend(tokens)

    tokens_np = np.array(all_tokens, dtype=np.uint32)

    assert tokens_np.max() < 2**16, "token id exceeds uint16 range"
    return tokens_np.astype(np.uint16)


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir',  type=str, default='../bioG_data')
    parser.add_argument('--out_dir', type=str, default='../bioG_tokens')
    parser.add_argument('--files',   type=str, nargs='+', default=None,
                        help='Specific .txt filenames to tokenize')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = args.files if args.files else [
        'bioG_train.txt',
        'bioG_val.txt',
    ]

    for fname in files:
        in_path  = os.path.join(args.in_dir,  fname)
        out_path = os.path.join(args.out_dir, fname.replace('.txt', '.npy'))

        if not os.path.exists(in_path):
            print(f"[skip] {in_path} not found")
            continue

        print(f"\n{fname}")
        tokens = tokenize_file(in_path)
        np.save(out_path, tokens)

        print(f"  {len(tokens):,} tokens → {out_path}  ({tokens.nbytes / 1e6:.1f} MB)")


if __name__ == '__main__':
    main()