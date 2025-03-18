import os
import pandas as pd
from datasets import load_dataset

def load_custom_dataset(file_path):
    """
    Loads a custom dataset from TXT, CSV, or JSON format.

    Args:
    - file_path (str): Path to the dataset file.

    Returns:
    - dataset (list): A list of text samples.
    """
    file_ext = file_path.split(".")[-1]

    if file_ext == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.readlines()
        return [{"text": line.strip()} for line in data if line.strip()]

    elif file_ext == "csv":
        df = pd.read_csv(file_path)
        return [{"text": row} for row in df.iloc[:, 0].dropna().tolist()]

    elif file_ext == "json":
        df = pd.read_json(file_path)
        return [{"text": row} for row in df.iloc[:, 0].dropna().tolist()]

    else:
        raise ValueError("Unsupported file format. Please use TXT, CSV, or JSON.")

def get_dataset(dataset_name=None, dataset_path=None, model_name=None, tokenizer=None):
    """
    Loads and preprocesses either a Hugging Face dataset or a custom dataset.

    Args:
    - dataset_name (str): The name of the dataset from Hugging Face.
    - dataset_path (str): Path to a custom dataset file.
    - model_name (str): The model to be trained (e.g., "gpt2" or "t5-small").
    - tokenizer: The tokenizer instance for the model.

    Returns:
    - train_dataset: Preprocessed training dataset.
    - val_dataset: Preprocessed validation dataset.
    """

    if dataset_path:
        print(f"📌 Loading custom dataset from {dataset_path}...")
        dataset = load_custom_dataset(dataset_path)
    elif dataset_name:
        print(f"📌 Downloading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, "wikitext-2-raw-v1")  # Default config
    else:
        raise ValueError("You must provide either `dataset_name` or `dataset_path`.")

    # GPT-2 doesn't have a padding token by default, so we set it to EOS token
    if model_name and "gpt2" in model_name:
        tokenizer.pad_token = tokenizer.eos_token  

    def preprocess(example):
        """
        Tokenizes the dataset.
        - Truncates text to fit the max model input size.
        - Pads sequences to the max length.
        - Adds labels (needed for GPT-2 loss calculation).
        """
        encoding = tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
        encoding["labels"] = encoding["input_ids"].copy()  # GPT-2 needs labels
        return encoding

    dataset = dataset.map(preprocess, batched=True)
    return dataset["train"], dataset["validation"]
