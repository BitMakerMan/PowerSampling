"""
Batch Processing Example with Power Sampling

This script demonstrates how to process multiple prompts efficiently
using the batch processing capabilities of Power Sampling.
"""

import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from power_sampling import load_model_and_tokenizer, PowerSampler


def main():
    """Demonstrate batch processing with power sampling."""

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model_name = "microsoft/DialoGPT-small"  # Using smaller model for batch processing

    try:
        model, tokenizer = load_model_and_tokenizer(model_name)
        print(f"Successfully loaded {model_name}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize power sampler
    sampler = PowerSampler(model, tokenizer)

    # Batch of prompts for processing
    prompts = [
        "What is artificial intelligence?",
        "How do vaccines work?",
        "Why is the sky blue?",
        "What causes earthquakes?",
        "How do plants make food?",
        "What is democracy?",
        "How does the internet work?",
        "Why do we dream?",
        "What is gravity?",
        "How do batteries store energy?"
    ]

    # Power sampling parameters for batch processing
    batch_params = {
        "alpha": 3.0,          # Slightly lower for speed in batch
        "block_size": 64,      # Smaller blocks for faster processing
        "steps": 3,            # Fewer steps for batch efficiency
        "max_len": 256,        # Shorter responses for batch demo
        "temperature": 0.9,
        "show_progress": True
    }

    print(f"\nProcessing {len(prompts)} prompts with Power Sampling...")
    print(f"Parameters: {batch_params}")
    print("="*60)

    # Process batch
    try:
        results = sampler.batch_power_sample(prompts, **batch_params)

        # Display results
        for i, (prompt, result) in enumerate(zip(prompts, results), 1):
            print(f"\n{i}. Prompt: {prompt}")
            print(f"   Response: {result[:150]}{'...' if len(result) > 150 else ''}")
            print("-" * 40)

    except Exception as e:
        print(f"Error in batch processing: {e}")

    print("\nBatch processing complete!")


if __name__ == "__main__":
    main()