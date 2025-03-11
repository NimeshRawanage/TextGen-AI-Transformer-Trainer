from datasets import load_dataset

def get_dataset(dataset_name, model_name, tokenizer):
    """
    Loads and preprocesses the dataset for training.
    Adds labels for GPT-2 to enable loss calculation.

    Args:
    - dataset_name (str): The name of the dataset from Hugging Face.
    - model_name (str): The model to be trained (e.g., "gpt2" or "t5-small").
    - tokenizer: The tokenizer instance for the model.

    Returns:
    - train_dataset: Preprocessed training dataset.
    - val_dataset: Preprocessed validation dataset.
    """

    print(f"📌 Downloading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, "wikitext-103-raw-v1")

    # GPT-2 doesn't have a padding token by default, so we set it to EOS token
    if "gpt2" in model_name:
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

    # Apply tokenization to both training & validation datasets
    dataset = dataset.map(preprocess, batched=True)
    return dataset["train"], dataset["validation"]
