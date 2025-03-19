import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from src.preprocess import get_dataset

import torch
import argparse

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model_name="gpt2", dataset_name=None, dataset_path=None, output_dir="models/", epochs=3, batch_size=4):
    """
    Trains a GPT-2 or T5 model using Hugging Face's Trainer API.

    Args:
    - model_name (str): Name of the model to fine-tune (default: "gpt2").
    - dataset_name (str): Name of the dataset from Hugging Face (if provided).
    - dataset_path (str): Path to a custom dataset (if provided).
    - output_dir (str): Directory where the trained model will be saved.
    - epochs (int): Number of training epochs.
    - batch_size (int): Batch size for training.
    """

    print(f"🔹 Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load pre-trained model
    if "t5" in model_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)  # T5 for summarization/paraphrasing
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)  # GPT-2 for text generation

    # Load and preprocess dataset using `preprocess.py`
    print(f"📌 Preparing dataset...")
    train_dataset, val_dataset = get_dataset(dataset_name, dataset_path, model_name, tokenizer)

    # ✅ Generate a unique model save directory
    dataset_id = dataset_name if dataset_name else os.path.basename(dataset_path).split('.')[0]
    model_save_dir = os.path.join(output_dir, f"{model_name}_{dataset_id}")
    os.makedirs(model_save_dir, exist_ok=True)  # Ensure directory exists

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=model_save_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        save_steps=500,
        save_total_limit=2,
        logging_dir="logs/",
        logging_steps=100,
        eval_strategy="epoch",
        report_to="none",
        fp16=True  # Enable mixed precision for faster training
    )

    # Define data collator (needed for GPT-2)
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
        data_collator=data_collator
    )

    print("🚀 Training started...")
    trainer.train()

    # ✅ Save the trained model in the unique directory
    model.save_pretrained(model_save_dir, safe_serialization=False)
    tokenizer.save_pretrained(model_save_dir)
    print(f"✅ Training complete! Model saved to {model_save_dir}")

if __name__ == "__main__":
    # Define command-line arguments for flexibility
    parser = argparse.ArgumentParser(description="Fine-Tune GPT-2 or T5 for Text Generation")
    parser.add_argument("--model", type=str, default="gpt2", help="Model to fine-tune (e.g., 'gpt2' or 't5-small')")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name from Hugging Face (e.g., 'wikitext')")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to a custom dataset file (TXT, CSV, JSON)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="models/", help="Where to save trained model")

    args = parser.parse_args()
    train(args.model, args.dataset, args.dataset_path, args.output_dir, args.epochs, args.batch_size)
