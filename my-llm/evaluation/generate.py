import argparse
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import load_model_from_config


class TextGenerator:
    """
    Simple GPT text generator.
    """

    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_path: str = "tokenizer/tokenizer.json",
        device: str = "cpu",
    ):

        self.device = torch.device(device)

        print("\nLoading tokenizer...")

        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        print("Tokenizer loaded.")

        print("\nCreating model...")

        self.model = load_model_from_config()

        print("Loading checkpoint...")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model"]
        )

        self.model.to(self.device)

        self.model.eval()

        print(
            f"Checkpoint Loaded (Step {checkpoint['step']})"
        )

    # --------------------------------------------------
    # Generate
    # --------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 40,
    ):

        encoding = self.tokenizer.encode(prompt)

        input_ids = torch.tensor(
            [encoding.ids],
            dtype=torch.long,
            device=self.device,
        )

        generated = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        output = self.tokenizer.decode(
            generated[0].tolist()
        )

        return output

    # --------------------------------------------------
    # Interactive Chat
    # --------------------------------------------------

    def chat(self):

        print("\n" + "=" * 60)
        print("GPT Text Generator")
        print("=" * 60)

        print("\nType 'exit' to quit.\n")

        while True:

            prompt = input("You : ").strip()

            if prompt.lower() in [
                "exit",
                "quit",
                "q",
            ]:
                break

            if not prompt:
                continue

            print("\nGenerating...\n")

            response = self.generate(prompt)

            print("GPT :", response)
            print()
# --------------------------------------------------
# Command Line Arguments
# --------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description="Generate text using a trained GPT model"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/checkpoint-final.pt",
        help="Path to checkpoint",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="tokenizer/tokenizer.json",
        help="Path to tokenizer",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use",
    )

    return parser.parse_args()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    args = parse_args()

    print("=" * 60)
    print("GPT Text Generation")
    print("=" * 60)

    if args.device == "auto":

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:

        device = args.device

    print(f"\nDevice : {device}")

    checkpoint = Path(args.checkpoint)

    if not checkpoint.exists():

        print(
            f"\nCheckpoint not found:\n{checkpoint}"
        )

        return

    tokenizer = Path(args.tokenizer)

    if not tokenizer.exists():

        print(
            f"\nTokenizer not found:\n{tokenizer}"
        )

        return

    generator = TextGenerator(
        checkpoint_path=str(checkpoint),
        tokenizer_path=str(tokenizer),
        device=device,
    )

    generator.chat()

    print("\nGoodbye!")


if __name__ == "__main__":
    main()