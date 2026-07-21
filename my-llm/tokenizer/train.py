#!/usr/bin/env python3
"""
Train a BPE Tokenizer

Flow:
Raw Text
    ↓
BPE Training
    ↓
tokenizer.json
"""

import argparse
import sys
from pathlib import Path
from typing import List

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, StripAccents, Sequence


# --------------------------------------------------------
# Load Dataset
# --------------------------------------------------------

def load_text_files(data_path: str) -> List[str]:
    """
    Load one text file or every .txt file inside a folder.
    """

    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"{data_path} not found.")

    if path.is_file():
        files = [str(path)]

    else:
        files = [str(file) for file in path.rglob("*.txt")]

    if not files:
        raise ValueError("No text files found.")

    print(f"Found {len(files)} file(s).")

    return files


# --------------------------------------------------------
# Train BPE Tokenizer
# --------------------------------------------------------

def train_tokenizer(
    files: List[str],
    vocab_size: int,
    min_frequency: int,
    special_tokens: List[str],
):

    print("\nTraining BPE tokenizer...\n")

    tokenizer = Tokenizer(
        BPE(
            unk_token="<unk>"
        )
    )

    tokenizer.normalizer = Sequence(
        [
            NFD(),
            StripAccents(),
        ]
    )

    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(

        vocab_size=vocab_size,

        min_frequency=min_frequency,

        special_tokens=special_tokens,

        show_progress=True,

    )

    tokenizer.train(files, trainer)

    return tokenizer


# --------------------------------------------------------
# Statistics
# --------------------------------------------------------

def show_statistics(
    tokenizer: Tokenizer,
    files: List[str],
):

    print("\n" + "=" * 60)

    print("Tokenizer Statistics")

    print("=" * 60)

    print(
        f"Vocabulary Size : {tokenizer.get_vocab_size():,}"
    )

    sample = Path(files[0]).read_text(
        encoding="utf-8"
    )[:200]

    encoded = tokenizer.encode(sample)

    print("\nSample Text\n")

    print(sample)

    print("\nTokens\n")

    print(encoded.tokens[:30])

    print("\nToken Count :", len(encoded.tokens))

    print("=" * 60)
# --------------------------------------------------------
# Main
# --------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer for the GPT model."
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Training text file or directory"
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8000,
        help="Vocabulary size (default: 8000)"
    )

    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum token frequency"
    )

    parser.add_argument(
        "--output",
        default="tokenizer/tokenizer.json",
        help="Output tokenizer path"
    )

    args = parser.parse_args()

    special_tokens = [
        "<pad>",
        "<unk>",
        "<bos>",
        "<eos>",
    ]

    try:

        print(f"\nLoading dataset from: {args.data}")

        files = load_text_files(args.data)

        total_size = sum(
            Path(file).stat().st_size
            for file in files
        )

        print(
            f"Dataset Size : {total_size / (1024 * 1024):.2f} MB"
        )

        tokenizer = train_tokenizer(

            files=files,

            vocab_size=args.vocab_size,

            min_frequency=args.min_frequency,

            special_tokens=special_tokens,

        )

        show_statistics(
            tokenizer,
            files,
        )

        output_path = Path(args.output)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        tokenizer.save(str(output_path))

        print("\nTokenizer saved successfully!")

        print(f"Location : {output_path}")

        print("\nNext Step")

        print("Prepare dataset")

        print("python data/prepare.py")

        print("\nThen train")

        print("python training/train.py")

    except Exception as e:

        print(f"\nError : {e}")

        sys.exit(1)


if __name__ == "__main__":

    main()