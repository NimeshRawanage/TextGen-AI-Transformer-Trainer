import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import argparse
import os

def load_model(model_dir):
    """
    Loads a trained model and tokenizer from the specified directory.

    Args:
    - model_dir (str): Path to the trained model directory.

    Returns:
    - model: The loaded Hugging Face model.
    - tokenizer: The tokenizer associated with the model.
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory '{model_dir}' not found.")

    print(f"Loading trained model from: {model_dir}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Load model based on the type (GPT-2 for causal generation, T5 for sequence-to-sequence tasks)
    if "t5" in model_dir:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir)

    # Set model to evaluation mode
    model.eval()
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=50):
    """
    Generates text using the trained model.

    Args:
    - model: Trained language model.
    - tokenizer: Tokenizer for the model.
    - prompt (str): Input text for generation.
    - max_length (int): Maximum length of generated output.

    Returns:
    - str: Generated text.
    """
    # Tokenize input text and generate attention mask
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).input_ids
    attention_mask = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).attention_mask

    # Generate output text
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, 
            attention_mask=attention_mask, 
            max_length=max_length, 
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode generated output to human-readable text
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Define command-line arguments for inference
    parser = argparse.ArgumentParser(description="Generate text using a fine-tuned model.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the trained model directory.")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for text generation.")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length of generated text.")

    args = parser.parse_args()

    # Load trained model and tokenizer
    model, tokenizer = load_model(args.model_dir)

    # Generate text based on the provided prompt
    generated_text = generate_text(model, tokenizer, args.prompt, args.max_length)
    
    print("\nGenerated Text:", generated_text)

