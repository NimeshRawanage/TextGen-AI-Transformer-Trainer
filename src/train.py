import os
import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling  # Required for GPT-2 loss calculation
)

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

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

def train(model_name="gpt2", dataset_name="wikitext", output_dir="models/", epochs=3, batch_size=4):
    """
    Trains a GPT-2 or T5 model using Hugging Face's Trainer API.

    Args:
    - model_name (str): Name of the model to fine-tune (default: "gpt2").
    - dataset_name (str): Name of the dataset from Hugging Face.
    - output_dir (str): Directory where the trained model will be saved.
    - epochs (int): Number of training epochs.
    - batch_size (int): Batch size for training.
    """

    print(f"🔹 Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load pre-trained model
    if "t5" in model_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)  # T5 model for summarization/paraphrasing
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)  # GPT-2 for text generation

    # Load and preprocess dataset
    print(f"📌 Preparing dataset: {dataset_name}")
    train_dataset, val_dataset = get_dataset(dataset_name, model_name, tokenizer)  # Get both training & validation sets

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,  # Where to save the model
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        save_steps=500,
        save_total_limit=2,
        logging_dir="logs/",
        logging_steps=100,
        eval_strategy="epoch",  # Fixed: Previously deprecated "evaluation_strategy"
        report_to="none"
    )

    # Define data collator (needed for causal language models like GPT-2)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # GPT-2 is not a masked language model (MLM)
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # Training dataset
        eval_dataset=val_dataset,  # Validation dataset
        data_collator=data_collator  # Required for GPT-2 loss computation
    )

    print("🚀 Training started...")
    trainer.train()

    # Save the trained model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Training complete! Model saved to {output_dir}")

if __name__ == "__main__":
    # Define command-line arguments for flexibility
    parser = argparse.ArgumentParser(description="Fine-Tune GPT-2 or T5 for Text Generation")
    parser.add_argument("--model", type=str, default="gpt2", help="Model to fine-tune (e.g., 'gpt2' or 't5-small')")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset name from Hugging Face")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="models/", help="Where to save trained model")

    args = parser.parse_args()
    train(args.model, args.dataset, args.output_dir, args.epochs, args.batch_size)
