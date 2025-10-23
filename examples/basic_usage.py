"""
Basic Usage Example of Power Sampling

This script demonstrates how to use Power Sampling to improve LLM reasoning
on a simple scientific explanation task.
"""

import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from power_sampling import load_model_and_tokenizer, power_sample


def main():
    """Main function demonstrating basic power sampling usage."""

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model_name = "Qwen/Qwen2.5-7B-Instruct"  # You can change this to any compatible model

    try:
        model, tokenizer = load_model_and_tokenizer(model_name)
        print(f"Successfully loaded {model_name}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have the model available or change the model_name")
        return

    # Example prompts
    prompts = [
        "Explain how quantum entanglement works in simple terms.",
        "What are the main causes of climate change and how can we address them?",
        "Describe the process of photosynthesis and its importance for life on Earth.",
        "How do neural networks learn to recognize patterns in data?"
    ]

    # Power sampling parameters (based on recommendations)
    alpha = 4.0      # Sharpening factor
    block_size = 192 # Block size for resampling
    steps = 10       # Number of Metropolis-Hastings iterations
    max_len = 512    # Maximum sequence length

    print("\n" + "="*60)
    print("POWER SAMPLING DEMONSTRATION")
    print("="*60)

    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Example {i} ---")
        print(f"Prompt: {prompt}")
        print("\nGenerating response with Power Sampling...")

        try:
            # Generate response using power sampling
            response = power_sample(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                alpha=alpha,
                block_size=block_size,
                steps=steps,
                max_len=max_len,
                show_progress=True
            )

            print(f"\nResponse:\n{response}")
            print("-" * 40)

        except Exception as e:
            print(f"Error generating response: {e}")

    print("\nDemonstration complete!")


if __name__ == "__main__":
    main()