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
    
    # Load tokenizer from the specified directory
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Load model (GPT-2 for causal text generation, T5 for sequence-to-sequence tasks)
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
    # Tokenize input text and create attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    # Generate output text
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, 
            attention_mask=attention_mask, 
            max_length=max_length, 
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode generated output into readable text
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def chat_with_model(model, tokenizer, max_length=100):
    """
    Interactive chat function that allows continuous user interaction.

    Args:
    - model: Trained language model.
    - tokenizer: Tokenizer for the model.
    - max_length (int): Maximum response length.
    """
    chat_history = []  # Stores conversation history

    print("Chatbot is ready! Type 'exit' to end the session.\n")

    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Append user input to chat history
        chat_history.append(f"User: {user_input}")

        # Format input as a conversation
        input_text = "\n".join(chat_history) + "\nBot:"
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        
        # Generate response
        with torch.no_grad():
            output_ids = model.generate(
                inputs.input_ids, 
                attention_mask=inputs.attention_mask, 
                max_length=max_length, 
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Extract the bot's response
        bot_reply = tokenizer.decode(output_ids[0], skip_special_tokens=True).split("Bot:")[-1].strip()
        print("Bot:", bot_reply)

        # Update chat history
        chat_history.append(f"Bot: {bot_reply}")

if __name__ == "__main__":
    # Define command-line arguments for inference
    parser = argparse.ArgumentParser(description="Generate text or chat using a fine-tuned model.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the trained model directory.")
    parser.add_argument("--mode", type=str, default="text", choices=["text", "chat"], help="Run in text generation or chat mode.")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt for text generation (only used in 'text' mode).")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length of generated text.")

    args = parser.parse_args()

    # Load trained model and tokenizer
    model, tokenizer = load_model(args.model_dir)

    if args.mode == "chat":
        chat_with_model(model, tokenizer, args.max_length)
    else:
        if not args.prompt:
            raise ValueError("Prompt is required in 'text' mode.")
        generated_text = generate_text(model, tokenizer, args.prompt, args.max_length)
        print("\nGenerated Text:", generated_text)

