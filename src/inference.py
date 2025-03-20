import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import sys

# Function to detect if the script is running inside Google Colab
def detect_colab():
    return "google.colab" in sys.modules

# Function to handle user input in both terminal and Google Colab environments
def get_user_input():
    """
    Handles user input dynamically based on the environment.
    Uses `input()` for terminal execution.
    Uses `IPython` input for Colab to allow interactive input.
    """
    if detect_colab():
        from IPython.display import display
        import IPython
        user_input = IPython.input("User: ")
    else:
        user_input = input("User: ")
    return user_input.strip()

def load_model(model_dir):
    """
    Loads a trained language model and tokenizer from the specified directory.

    Args:
    - model_dir (str): Path to the directory containing the trained model and tokenizer.

    Returns:
    - model (transformers.PreTrainedModel): Loaded model instance.
    - tokenizer (transformers.PreTrainedTokenizer): Associated tokenizer for text processing.
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory '{model_dir}' not found.")

    print(f"Loading trained model from: {model_dir}")

    # Load tokenizer from the specified directory
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Load model (GPT-2 for causal text generation)
    model = AutoModelForCausalLM.from_pretrained(model_dir)

    # Ensure the tokenizer has a padding token (GPT-2 does not have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()  # Set model to evaluation mode
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=50):
    """
    Generates text based on a given input prompt using the trained model.

    Args:
    - model (transformers.PreTrainedModel): Pretrained language model.
    - tokenizer (transformers.PreTrainedTokenizer): Tokenizer for text processing.
    - prompt (str): Input text prompt to guide text generation.
    - max_length (int): Maximum number of tokens to generate.

    Returns:
    - str: Generated text output.
    """
    # Tokenize input prompt and create an attention mask
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).input_ids
    attention_mask = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).attention_mask

    with torch.no_grad():
        # Generate text with improved settings to reduce repetition
        output_ids = model.generate(
            input_ids, 
            attention_mask=attention_mask,  # Ensures proper token handling
            max_length=max_length, 
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True, # Enable sampling for better responses
            temperature=0.8,  # Introduces randomness in responses
            top_p=0.9,  # Enables nucleus sampling for diverse responses
            repetition_penalty=1.5  # Reduces repetition in generated text
        )

    # Decode generated output into a human-readable string
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def chat_with_model(model, tokenizer, max_length=100):
    """
    Interactive chatbot session for real-time user interaction.

    Args:
    - model (transformers.PreTrainedModel): Pretrained language model.
    - tokenizer (transformers.PreTrainedTokenizer): Tokenizer for text processing.
    - max_length (int): Maximum number of tokens in generated responses.
    """
    print("Chatbot is ready! Type 'exit' to end the session.\n")

    while True:
        user_input = get_user_input()  # Get user input from CLI or Colab
        if user_input.lower() == "exit":
            print("Chatbot session ended.")
            break

        response = generate_text(model, tokenizer, user_input, max_length)
        print(f"Bot: {response}\n")

if __name__ == "__main__":
    # Command-line argument parser for flexible usage
    parser = argparse.ArgumentParser(description="Generate text using a fine-tuned language model.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the trained model directory.")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt for text generation (only required for 'generate' mode).")
    parser.add_argument("--mode", type=str, choices=["generate", "chat"], default="generate", help="Select mode: 'generate' for single prompt completion, 'chat' for interactive conversation.")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length of generated text.")

    args = parser.parse_args()

    # Load the trained model and tokenizer
    model, tokenizer = load_model(args.model_dir)

    # Run in chat mode or single-prompt mode based on user selection
    if args.mode == "chat":
        chat_with_model(model, tokenizer, args.max_length)
    else:
        if not args.prompt:
            raise ValueError("Prompt is required in 'generate' mode.")
        
        generated_text = generate_text(model, tokenizer, args.prompt, args.max_length)
        print("\nGenerated Text:", generated_text)


