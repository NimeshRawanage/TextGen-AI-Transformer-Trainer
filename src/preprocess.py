import os
import pandas as pd
from datasets import Dataset, load_dataset

def load_custom_dataset(file_path):
    """
    Loads a custom dataset from a TXT, CSV, or JSON file.

    Args:
    - file_path (str): Path to the dataset file.

    Returns:
    - dataset (Dataset): Hugging Face Dataset object containing text samples.
    """
    file_ext = file_path.split(".")[-1]

    if file_ext == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            data = [{"text": line.strip()} for line in f.readlines() if line.strip()]

    elif file_ext == "csv":
        df = pd.read_csv(file_path)
        data = [{"text": row} for row in df.iloc[:, 0].dropna().tolist()]

    elif file_ext == "json":
        df = pd.read_json(file_path)
        data = [{"text": row} for row in df.iloc[:, 0].dropna().tolist()]

    else:
        raise ValueError("Unsupported file format. Please use TXT, CSV, or JSON.")

    return Dataset.from_list(data)  # Convert to Hugging Face Dataset format

def get_dataset(dataset_name=None, dataset_path=None, model_name=None, tokenizer=None):
    """
    Loads and preprocesses either a Hugging Face dataset or a custom dataset.

    Args:
    - dataset_name (str): The name of the dataset from Hugging Face.
    - dataset_path (str): Path to a custom dataset file.
    - model_name (str): The model being trained (e.g., "gpt2" or "t5-small").
    - tokenizer: The tokenizer instance for the model.

    Returns:
    - train_dataset: Preprocessed training dataset.
    - val_dataset: Preprocessed validation dataset.
    """

    if dataset_path:
        print(f"Loading custom dataset from {dataset_path}...")
        dataset = load_custom_dataset(dataset_path)

        # Convert dataset to Hugging Face Dataset format
        dataset = Dataset.from_list(dataset)

        # Split dataset: 90% training, 10% validation
        dataset = dataset.train_test_split(test_size=0.1)
        train_dataset = dataset["train"]
        val_dataset = dataset["test"]

    elif dataset_name:
        print(f"Downloading dataset: {dataset_name}...")
        dataset = load_dataset(dataset_name, "wikitext-2-raw-v1")  # Default config

        # Hugging Face datasets return a DatasetDict with predefined splits
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]

    else:
        raise ValueError("You must provide either `dataset_name` or `dataset_path`.")

    # Set padding token for GPT-2, as it doesn't have one by default
    if model_name and "gpt2" in model_name:
        tokenizer.pad_token = tokenizer.eos_token  

    def preprocess(example):
        """
        Tokenizes the dataset using the specified tokenizer.

        - Truncates text to fit the maximum model input size.
        - Pads sequences to the maximum length.
        - Adds labels (required for GPT-2 loss calculation).
        """
        encoding = tokenizer(
            example["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=512
        )
        encoding["labels"] = encoding["input_ids"].copy()  # GPT-2 needs labels
        return encoding

    # Apply tokenization to datasets
    train_dataset = train_dataset.map(preprocess, batched=True)
    val_dataset = val_dataset.map(preprocess, batched=True)

    return train_dataset, val_dataset

