"""
PyTorch Dataset for GPT Training

Loads train.pt / val.pt and prepares batches for training.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


class LLMDataset(Dataset):
    """
    Dataset for GPT training.
    """

    def __init__(
        self,
        data_path: str,
        max_length: int = 512,
        pad_token_id: int = 0,
    ):

        self.data_path = Path(data_path)
        self.max_length = max_length
        self.pad_token_id = pad_token_id

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"{self.data_path} not found."
            )

        self.data = torch.load(self.data_path)

        if not isinstance(self.data, torch.Tensor):
            raise ValueError(
                "Dataset must be a PyTorch tensor."
            )

        if self.data.dim() != 2:
            raise ValueError(
                "Dataset must have shape [num_examples, sequence_length]."
            )

        print(
            f"Loaded {len(self.data):,} training examples."
        )

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        sequence = self.data[index]

        if len(sequence) > self.max_length:

            sequence = sequence[: self.max_length]

        seq_length = len(sequence)

        if seq_length < self.max_length:

            padding = torch.full(

                (self.max_length - seq_length,),

                self.pad_token_id,

                dtype=torch.long,

            )

            sequence = torch.cat(
                [
                    sequence,
                    padding,
                ]
            )

        attention_mask = torch.ones(
            self.max_length,
            dtype=torch.long,
        )

        if seq_length < self.max_length:

            attention_mask[seq_length:] = 0

        labels = sequence.clone()

        labels[attention_mask == 0] = -100

        return {

            "input_ids": sequence,

            "attention_mask": attention_mask,

            "labels": labels,

        }
    # --------------------------------------------------------
# DataLoader
# --------------------------------------------------------

def create_dataloader(
    dataset: LLMDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    """
    Create a PyTorch DataLoader.
    """

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,
    )


# --------------------------------------------------------
# Test
# --------------------------------------------------------

if __name__ == "__main__":

    print("Testing Dataset...\n")

    # Create dummy data
    dummy = torch.randint(
        0,
        1000,
        (100, 512),
    )

    torch.save(
        dummy,
        "test.pt",
    )

    dataset = LLMDataset(
        "test.pt",
        max_length=512,
    )

    print(f"Dataset Size : {len(dataset)}")

    sample = dataset[0]

    print("\nSample")

    print("Input IDs Shape :", sample["input_ids"].shape)
    print("Attention Mask :", sample["attention_mask"].shape)
    print("Labels Shape :", sample["labels"].shape)

    dataloader = create_dataloader(
        dataset,
        batch_size=4,
    )

    batch = next(iter(dataloader))

    print("\nBatch")

    print("Input IDs :", batch["input_ids"].shape)
    print("Attention Mask :", batch["attention_mask"].shape)
    print("Labels :", batch["labels"].shape)

    Path("test.pt").unlink()

    print("\nDataset test completed successfully.")