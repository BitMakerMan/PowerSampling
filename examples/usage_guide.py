"""
Complete Usage Guide for Power Sampling
This file shows all the ways you can use Power Sampling
"""

import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from power_sampling import load_model_and_tokenizer, power_sample, PowerSampler


def example_1_basic_usage():
    """Example 1: Basic single usage"""
    print("=== Example 1: Basic Usage ===")

    # Load a small model for testing
    model_name = "EleutherAI/gpt-neo-125M"
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Simple power sampling
    prompt = "Explain photosynthesis in simple terms:"
    response = power_sample(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        alpha=4.0,          # Sharpening factor (higher = more focused)
        block_size=64,      # Block size for resampling
        steps=5,            # Number of iterations
        max_len=200,        # Maximum response length
        show_progress=True
    )

    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print()


def example_2_class_usage():
    """Example 2: Using the PowerSampler class directly"""
    print("=== Example 2: Class Usage ===")

    model_name = "EleutherAI/gpt-neo-125M"
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Create sampler instance
    sampler = PowerSampler(model, tokenizer)

    # Use multiple prompts
    prompts = [
        "What is artificial intelligence?",
        "How does the internet work?",
        "Explain gravity in simple terms."
    ]

    for prompt in prompts:
        response = sampler.power_sample(
            prompt=prompt,
            alpha=3.0,
            block_size=48,
            steps=3,
            max_len=150,
            show_progress=False
        )
        print(f"Q: {prompt}")
        print(f"A: {response[:100]}...")
        print("-" * 50)


def example_3_batch_processing():
    """Example 3: Batch processing multiple prompts"""
    print("=== Example 3: Batch Processing ===")

    model_name = "EleutherAI/gpt-neo-125M"
    model, tokenizer = load_model_and_tokenizer(model_name)

    sampler = PowerSampler(model, tokenizer)

    # Multiple prompts for batch processing
    prompts = [
        "What is machine learning?",
        "How do vaccines work?",
        "Why is the sky blue?"
    ]

    # Process all prompts at once
    results = sampler.batch_power_sample(
        prompts=prompts,
        alpha=2.5,
        block_size=32,
        steps=2,
        max_len=120,
        show_progress=True
    )

    for i, (prompt, response) in enumerate(zip(prompts, results)):
        print(f"{i+1}. Q: {prompt}")
        print(f"   A: {response[:80]}...")
        print()


def example_4_parameter_tuning():
    """Example 4: Different parameter settings"""
    print("=== Example 4: Parameter Tuning ===")

    model_name = "EleutherAI/gpt-neo-125M"
    model, tokenizer = load_model_and_tokenizer(model_name)

    prompt = "What is climate change?"

    # Different parameter combinations
    configs = [
        {"alpha": 2.0, "block_size": 32, "steps": 2, "name": "Fast/Low Quality"},
        {"alpha": 4.0, "block_size": 64, "steps": 5, "name": "Balanced"},
        {"alpha": 6.0, "block_size": 96, "steps": 8, "name": "Slow/High Quality"},
    ]

    for config in configs:
        print(f"\n--- {config['name']} ---")
        response = power_sample(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            alpha=config["alpha"],
            block_size=config["block_size"],
            steps=config["steps"],
            max_len=100,
            show_progress=True
        )
        print(f"Response: {response[:150]}...")


def example_5_different_models():
    """Example 5: Using different models"""
    print("=== Example 5: Different Models ===")

    # List of models to try (from smallest to largest)
    models = [
        "EleutherAI/gpt-neo-125M",      # 125M - very fast
        "EleutherAI/gpt-neo-1.3B",      # 1.3B - good balance
        # "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 1.1B - conversational
    ]

    prompt = "What is Python programming?"

    for model_name in models:
        try:
            print(f"\n--- Testing {model_name} ---")
            model, tokenizer = load_model_and_tokenizer(model_name)

            response = power_sample(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                alpha=3.0,
                block_size=48,
                steps=3,
                max_len=120,
                show_progress=False
            )

            print(f"Response: {response[:100]}...")

        except Exception as e:
            print(f"Error with {model_name}: {e}")


def main():
    """Run all examples"""
    print("Power Sampling - Complete Usage Guide")
    print("=" * 50)

    try:
        # Run examples (comment out ones you don't want to run)
        example_1_basic_usage()
        example_2_class_usage()
        example_3_batch_processing()
        example_4_parameter_tuning()
        example_5_different_models()

    except KeyboardInterrupt:
        print("\nUsage guide interrupted by user")
    except Exception as e:
        print(f"Error in usage guide: {e}")


if __name__ == "__main__":
    main()