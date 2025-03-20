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

# Check if GPU is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model_name="gpt2", dataset_name=None, dataset_path=None, output_dir="models/", epochs=3, batch_size=4):
    """
    Fine-tunes a GPT-2 or T5 model using Hugging Face's Trainer API.

    Args:
    - model_name (str): Name of the pre-trained model to fine-tune (e.g., "gpt2" or "t5-small").
    - dataset_name (str, optional): Name of the dataset from Hugging Face (if provided).
    - dataset_path (str, optional): Path to a local dataset file (TXT, CSV, JSON).
    - output_dir (str): Directory to save the trained model.
    - epochs (int): Number of training epochs.
    - batch_size (int): Training batch size.
    """

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load pre-trained model based on its type (T5 for sequence-to-sequence tasks, GPT-2 for text generation)
    if "t5" in model_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Load and preprocess dataset
    print(f"Preparing dataset...")
    train_dataset, val_dataset = get_dataset(dataset_name, dataset_path, model_name, tokenizer)

    # Generate a unique directory name based on the model and dataset used
    dataset_id = dataset_name if dataset_name else os.path.basename(dataset_path).split('.')[0]
    model_save_dir = os.path.join(output_dir, f"{model_name}_{dataset_id}")
    os.makedirs(model_save_dir, exist_ok=True)

    # Define training configurations
    training_args = TrainingArguments(
        output_dir=model_save_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        save_steps=500,  # Save model checkpoint every 500 steps
        save_total_limit=2,  # Keep the last 2 model checkpoints
        logging_dir="logs/",
        logging_steps=100,  # Log training status every 100 steps
        eval_strategy="epoch",  # Run evaluation after each epoch
        report_to="none",  # Disable default reporting to Hugging Face Hub
        fp16=True  # Enable mixed precision training for improved performance
    )

    # Define a data collator for batch processing
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Set to False as GPT-2 is not a masked language model
    )

    # Initialize Hugging Face Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    print("Training started...")
    trainer.train()

    # Save the fine-tuned model and tokenizer
    model.save_pretrained(model_save_dir, safe_serialization=False)
    tokenizer.save_pretrained(model_save_dir)
    print(f"Training complete! Model saved to {model_save_dir}")

if __name__ == "__main__":
    # Define command-line arguments for running the script
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 or T5 for text generation tasks.")
    parser.add_argument("--model", type=str, default="gpt2", help="Pre-trained model to fine-tune (e.g., 'gpt2', 't5-small').")
    parser.add_argument("--dataset", type=str, default=None, help="Hugging Face dataset name (e.g., 'wikitext').")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to a local dataset file (TXT, CSV, JSON).")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--output_dir", type=str, default="models/", help="Directory to save the trained model.")

    # Parse command-line arguments and initiate training
    args = parser.parse_args()
    train(args.model, args.dataset, args.dataset_path, args.output_dir, args.epochs, args.batch_size)
