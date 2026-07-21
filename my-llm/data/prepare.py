#!/usr/bin/env python3
"""
Prepare text data for GPT training.

Pipeline:
Raw Text
    ↓
Tokenizer
    ↓
Token IDs
    ↓
Sliding Window
    ↓
Train / Validation Split
    ↓
train.pt / val.pt
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from tokenizers import Tokenizer


class DataPreprocessor:

    def __init__(
        self,
        tokenizer_path: str,
        max_length: int = 512,
        stride: int = 256,
    ):

        self.tokenizer = self.load_tokenizer(tokenizer_path)
        self.max_length = max_length
        self.stride = stride

    # -------------------------------------------------------
    # Load Tokenizer
    # -------------------------------------------------------

    def load_tokenizer(self, tokenizer_path: str):

        tokenizer_path = Path(tokenizer_path)

        if not tokenizer_path.exists():

            raise FileNotFoundError(
                f"Tokenizer not found : {tokenizer_path}"
            )

        return Tokenizer.from_file(str(tokenizer_path))

    # -------------------------------------------------------
    # Load Dataset
    # -------------------------------------------------------

    def load_text_files(self, data_path: str) -> str:

        path = Path(data_path)

        if not path.exists():
            raise FileNotFoundError(data_path)

        texts = []

        if path.is_file():

            texts.append(
                path.read_text(encoding="utf-8")
            )

        else:

            files = sorted(path.rglob("*.txt"))

            if not files:
                raise ValueError("No .txt files found.")

            for file in files:

                texts.append(
                    file.read_text(
                        encoding="utf-8"
                    )
                )

        return "\n".join(texts)

    # -------------------------------------------------------
    # Tokenize
    # -------------------------------------------------------

    def create_examples(
        self,
        text: str,
    ) -> List[List[int]]:

        encoding = self.tokenizer.encode(text)

        token_ids = encoding.ids

        print(
            f"Total Tokens : {len(token_ids):,}"
        )

        examples = []

        for start in range(

            0,

            len(token_ids) - self.max_length + 1,

            self.stride,

        ):

            sample = token_ids[
                start:
                start + self.max_length
            ]

            if len(sample) == self.max_length:

                examples.append(sample)

        if len(token_ids) > self.max_length:

            final = token_ids[-self.max_length:]

            if not examples or examples[-1] != final:

                examples.append(final)

        return examples

    # -------------------------------------------------------
    # Train / Validation Split
    # -------------------------------------------------------

    def split_dataset(

        self,

        examples: List[List[int]],

        validation_split: float = 0.1,

    ) -> Tuple[List[List[int]], List[List[int]]]:

        validation_size = int(

            len(examples) * validation_split

        )

        train_size = len(examples) - validation_size

        train_examples = examples[:train_size]

        validation_examples = examples[train_size:]

        return train_examples, validation_examples

    # -------------------------------------------------------
    # Save Dataset
    # -------------------------------------------------------

    def save_dataset(

        self,

        examples: List[List[int]],

        output_path: str,

    ):

        output = Path(output_path)

        output.parent.mkdir(

            parents=True,

            exist_ok=True,

        )

        tensor = torch.tensor(

            examples,

            dtype=torch.long,

        )

        torch.save(

            tensor,

            output,

        )

        print(

            f"Saved {len(examples):,} samples -> {output}"

        )
#!/usr/bin/env python3
"""
Prepare text data for GPT training.

Pipeline:
Raw Text
    ↓
Tokenizer
    ↓
Token IDs
    ↓
Sliding Window
    ↓
Train / Validation Split
    ↓
train.pt / val.pt
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from tokenizers import Tokenizer


class DataPreprocessor:

    def __init__(
        self,
        tokenizer_path: str,
        max_length: int = 512,
        stride: int = 256,
    ):

        self.tokenizer = self.load_tokenizer(tokenizer_path)
        self.max_length = max_length
        self.stride = stride

    # -------------------------------------------------------
    # Load Tokenizer
    # -------------------------------------------------------

    def load_tokenizer(self, tokenizer_path: str):

        tokenizer_path = Path(tokenizer_path)

        if not tokenizer_path.exists():

            raise FileNotFoundError(
                f"Tokenizer not found : {tokenizer_path}"
            )

        return Tokenizer.from_file(str(tokenizer_path))

    # -------------------------------------------------------
    # Load Dataset
    # -------------------------------------------------------

    def load_text_files(self, data_path: str) -> str:

        path = Path(data_path)

        if not path.exists():
            raise FileNotFoundError(data_path)

        texts = []

        if path.is_file():

            texts.append(
                path.read_text(encoding="utf-8")
            )

        else:

            files = sorted(path.rglob("*.txt"))

            if not files:
                raise ValueError("No .txt files found.")

            for file in files:

                texts.append(
                    file.read_text(
                        encoding="utf-8"
                    )
                )

        return "\n".join(texts)

    # -------------------------------------------------------
    # Tokenize
    # -------------------------------------------------------

    def create_examples(
        self,
        text: str,
    ) -> List[List[int]]:

        encoding = self.tokenizer.encode(text)

        token_ids = encoding.ids

        print(
            f"Total Tokens : {len(token_ids):,}"
        )

        examples = []

        for start in range(

            0,

            len(token_ids) - self.max_length + 1,

            self.stride,

        ):

            sample = token_ids[
                start:
                start + self.max_length
            ]

            if len(sample) == self.max_length:

                examples.append(sample)

        if len(token_ids) > self.max_length:

            final = token_ids[-self.max_length:]

            if not examples or examples[-1] != final:

                examples.append(final)

        return examples

    # -------------------------------------------------------
    # Train / Validation Split
    # -------------------------------------------------------

    def split_dataset(

        self,

        examples: List[List[int]],

        validation_split: float = 0.1,

    ) -> Tuple[List[List[int]], List[List[int]]]:

        validation_size = int(

            len(examples) * validation_split

        )

        train_size = len(examples) - validation_size

        train_examples = examples[:train_size]

        validation_examples = examples[train_size:]

        return train_examples, validation_examples

    # -------------------------------------------------------
    # Save Dataset
    # -------------------------------------------------------

    def save_dataset(

        self,

        examples: List[List[int]],

        output_path: str,

    ):

        output = Path(output_path)

        output.parent.mkdir(

            parents=True,

            exist_ok=True,

        )

        tensor = torch.tensor(

            examples,

            dtype=torch.long,

        )

        torch.save(

            tensor,

            output,

        )

        print(

            f"Saved {len(examples):,} samples -> {output}"

        )
    # -------------------------------------------------------
    # Statistics
    # -------------------------------------------------------

    def show_statistics(
        self,
        train_examples: List[List[int]],
        validation_examples: List[List[int]],
    ):

        train_tokens = len(train_examples) * self.max_length
        validation_tokens = len(validation_examples) * self.max_length

        print("\n" + "=" * 60)
        print("Dataset Statistics")
        print("=" * 60)

        print(f"Vocabulary Size     : {self.tokenizer.get_vocab_size():,}")
        print(f"Sequence Length     : {self.max_length}")
        print(f"Stride              : {self.stride}")

        print(f"\nTraining Samples    : {len(train_examples):,}")
        print(f"Validation Samples  : {len(validation_examples):,}")

        print(f"\nTraining Tokens     : {train_tokens:,}")
        print(f"Validation Tokens   : {validation_tokens:,}")
        print(f"Total Tokens        : {train_tokens + validation_tokens:,}")

        print("=" * 60)


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Prepare data for GPT training."
    )

    parser.add_argument(
        "--input",
        default="data/raw/",
        help="Input text file or folder",
    )

    parser.add_argument(
        "--tokenizer",
        default="tokenizer/tokenizer.json",
        help="Tokenizer path",
    )

    parser.add_argument(
        "--output-dir",
        default="data/processed/",
        help="Output directory",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Sequence length",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=256,
        help="Sliding window stride",
    )

    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split",
    )

    args = parser.parse_args()

    try:

        print("\nLoading tokenizer...")

        processor = DataPreprocessor(
            tokenizer_path=args.tokenizer,
            max_length=args.max_length,
            stride=args.stride,
        )

        print("Loading dataset...")

        text = processor.load_text_files(
            args.input
        )

        print(
            f"Characters Loaded : {len(text):,}"
        )

        print("\nCreating training examples...")

        examples = processor.create_examples(
            text
        )

        print(
            f"Examples Created : {len(examples):,}"
        )

        train_examples, validation_examples = processor.split_dataset(
            examples,
            args.val_split,
        )

        processor.save_dataset(
            train_examples,
            f"{args.output_dir}/train.pt",
        )

        if validation_examples:

            processor.save_dataset(
                validation_examples,
                f"{args.output_dir}/val.pt",
            )

        processor.show_statistics(
            train_examples,
            validation_examples,
        )

        print("\nData preparation completed successfully.")

        print("\nNext Step")

        print("python training/train.py")

    except Exception as e:

        print(f"\nError: {e}")

        sys.exit(1)


if __name__ == "__main__":

    main()