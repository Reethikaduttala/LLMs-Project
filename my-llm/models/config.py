"""
Model Configuration

Loads and validates llm.config.js
Creates a GPT model using the configuration.
"""

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from .gpt import create_gpt_model


class ConfigValidationError(Exception):
    """Raised when configuration is invalid."""
    pass


class ConfigLoader:
    """
    Loads and validates llm.config.js.
    """

    def __init__(self, config_path: str = "llm.config.js"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate()

    # ------------------------------------------------------------------
    # Load JavaScript configuration
    # ------------------------------------------------------------------

    def _load_config(self) -> Dict[str, Any]:

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )

        js = f"""
        const config=require('./{self.config_path}');
        console.log(JSON.stringify(config));
        """

        try:
            result = subprocess.run(
                ["node", "-e", js],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.config_path.parent
            )

            return json.loads(result.stdout)

        except FileNotFoundError:
            raise RuntimeError(
                "Node.js is required to load llm.config.js"
            )

        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Error while loading configuration:\n{e.stderr}"
            )

        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Invalid JSON returned from llm.config.js\n{e}"
            )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self):

        self._validate_model()
        self._validate_training()
        self._validate_data()
        self._validate_tokenizer()

    def _validate_model(self):

        model = self.config.get("model")

        if model is None:
            raise ConfigValidationError("Missing model configuration")

        required = [
            "vocab_size",
            "max_length",
            "layers",
            "heads",
            "dim"
        ]

        for key in required:

            if key not in model:
                raise ConfigValidationError(
                    f"Missing model.{key}"
                )

        if model["dim"] % model["heads"] != 0:
            raise ConfigValidationError(
                "Embedding dimension must be divisible by number of heads."
            )

        if not 0 <= model.get("dropout", 0.1) < 1:
            raise ConfigValidationError(
                "Dropout must be between 0 and 1."
            )

    def _validate_training(self):

        training = self.config.get("training")

        if training is None:
            raise ConfigValidationError(
                "Missing training configuration."
            )

        positive = [
            "batch_size",
            "learning_rate",
            "max_steps",
            "eval_interval",
            "save_interval"
        ]

        for key in positive:

            if training.get(key, 0) <= 0:
                raise ConfigValidationError(
                    f"training.{key} must be positive."
                )

    def _validate_data(self):

        data = self.config.get("data")

        if data is None:
            raise ConfigValidationError(
                "Missing data configuration."
            )

        if data["max_length"] <= 0:
            raise ConfigValidationError(
                "data.max_length must be positive."
            )

        if data["stride"] <= 0:
            raise ConfigValidationError(
                "data.stride must be positive."
            )

        if data["stride"] > data["max_length"]:
            raise ConfigValidationError(
                "Stride cannot exceed max_length."
            )

    def _validate_tokenizer(self):

        tokenizer = self.config.get("tokenizer")

        if tokenizer is None:
            raise ConfigValidationError(
                "Missing tokenizer configuration."
            )

        if tokenizer["vocab_size"] <= 0:
            raise ConfigValidationError(
                "Tokenizer vocab_size must be positive."
            )

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get(self, key: str, default=None):

        value = self.config

        for part in key.split("."):

            if isinstance(value, dict):
                value = value.get(part)
            else:
                return default

            if value is None:
                return default

        return value

    def model(self):
        return self.config["model"]

    def training(self):
        return self.config["training"]

    def data(self):
        return self.config["data"]

    def tokenizer(self):
        return self.config["tokenizer"]
# ------------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------------

def get_tokenizer_vocab_size(tokenizer_path: Path) -> Optional[int]:
    """
    Returns the vocabulary size from tokenizer/tokenizer.json.
    """

    if not tokenizer_path.exists():
        return None

    try:

        with open(tokenizer_path, "r", encoding="utf-8") as f:
            tokenizer = json.load(f)

        return len(tokenizer["model"]["vocab"])

    except Exception as e:
        raise ValueError(f"Unable to read tokenizer: {e}")


# ------------------------------------------------------------------
# Model Loader
# ------------------------------------------------------------------

def load_model_from_config(config_path: str = "llm.config.js"):
    """
    Loads the GPT model using llm.config.js.
    Automatically synchronizes vocab_size with tokenizer.json.
    """

    config = ConfigLoader(config_path)

    model_config = config.model().copy()

    tokenizer_path = Path("tokenizer/tokenizer.json")

    tokenizer_vocab = get_tokenizer_vocab_size(tokenizer_path)

    if tokenizer_vocab is not None:

        config_vocab = model_config["vocab_size"]

        if tokenizer_vocab != config_vocab:

            print(
                f"\nVocabulary mismatch detected.\n"
                f"Config : {config_vocab}\n"
                f"Tokenizer : {tokenizer_vocab}\n"
                f"Using tokenizer vocabulary."
            )

            model_config["vocab_size"] = tokenizer_vocab

        else:

            print(f"Vocabulary Size : {tokenizer_vocab}")

    else:

        print(
            "Tokenizer not found.\n"
            f"Using vocab_size={model_config['vocab_size']}"
        )

    model = create_gpt_model(model_config)

    print(
        f"Model Parameters : {model.count_parameters():,}"
    )

    return model


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":

    try:

        config = ConfigLoader()

        print("\nConfiguration Loaded Successfully\n")

        print("Model Configuration")
        print("-------------------")

        for key, value in config.model().items():
            print(f"{key:15}: {value}")

        print("\nTraining Configuration")
        print("----------------------")

        for key, value in config.training().items():
            print(f"{key:25}: {value}")

        model = load_model_from_config()

        print("\nModel created successfully.")

    except Exception as e:

        print("\nConfiguration Error")
        print("-------------------")
        print(e)