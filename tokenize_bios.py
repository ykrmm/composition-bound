
import os
import argparse
import numpy as np
import tiktoken
from tqdm import tqdm

EOS_STR = '<|endoftext|>'


def tokenize_file(path: str, enc: tiktoken.Encoding) -> np.ndarray:
    eot = enc._special_tokens[EOS_STR]

    with open(path, 'r') as f:
        raw = f.read()

    # Split on EOS separator, drop empty strings
    bios = [b.strip() for b in raw.split(EOS_STR) if b.strip()]
    print(f"  {len(bios):,} biographies found")

    all_tokens = []
    for bio in tqdm(bios, desc="  tokenizing", unit="bio"):
        tokens = [eot] + enc.encode_ordinary(bio)
        all_tokens.extend(tokens)

    tokens_np = np.array(all_tokens, dtype=np.uint32)
    assert tokens_np.max() < 2**16, "token id exceeds uint16 range"
    return tokens_np.astype(np.uint16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir',  type=str, default='bios_data')
    parser.add_argument('--out_dir', type=str, default='bios_tokens')
    parser.add_argument('--files',   type=str, nargs='+', default=None,
                        help='Specific .txt filenames to tokenize (default: all standard variants)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")

    files = args.files if args.files else [
        'bios_single_train.txt',
        'bios_single_val.txt',
        'bios_multi5p_train.txt',
        'bios_multi5p_val.txt',
    ]

    for fname in files:
        in_path  = os.path.join(args.in_dir,  fname)
        out_path = os.path.join(args.out_dir, fname.replace('.txt', '.npy'))

        if not os.path.exists(in_path):
            print(f"[skip] {in_path} not found")
            continue

        print(f"\n{fname}")
        tokens = tokenize_file(in_path, enc)
        np.save(out_path, tokens)
        print(f"  {len(tokens):,} tokens → {out_path}  ({tokens.nbytes / 1e6:.1f} MB)")


if __name__ == '__main__':
    main()
