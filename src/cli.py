"""
Command line interface for Power Sampling
"""

from .power_sampling import load_model_and_tokenizer, power_sample


def main():
    """Main CLI function for Power Sampling."""

    print("=== Power Sampling CLI ===")

    # Use smallest model for quick testing
    model_name = "EleutherAI/gpt-neo-125M"  # 125M parameters - very small

    print(f"Testing with model: {model_name}")
    print("Loading model...")

    try:
        model, tokenizer = load_model_and_tokenizer(model_name)
        print("[SUCCESS] Model loaded successfully!")

        # Simple test
        prompt = "What is machine learning?"
        print(f"\nPrompt: {prompt}")
        print("Running Power Sampling...")

        response = power_sample(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            alpha=2.0,      # Lower for faster testing
            block_size=32,  # Smaller blocks
            steps=2,        # Few iterations
            max_len=128,    # Short output
            show_progress=True
        )

        print(f"\nGenerated response:")
        print(response)
        print("\n[SUCCESS] Power Sampling test completed!")

    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())