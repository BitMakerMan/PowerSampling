"""
Power Sampling vs Standard Generation Comparison

This script compares outputs from standard generation vs power sampling
to demonstrate the improvements in reasoning and coherence.
"""

import sys
import os
import time

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from power_sampling import load_model_and_tokenizer, power_sample


def standard_generate(model, tokenizer, prompt, max_length=512, temperature=1.0):
    """Generate text using standard sampling."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length - len(inputs[0]),
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )

    # Extract only the generated part
    prompt_length = inputs['input_ids'].size(-1)
    generated_tokens = outputs[0][prompt_length:]

    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def main():
    """Compare standard generation with power sampling."""

    import torch  # Import here to avoid issues if torch not available

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model_name = "microsoft/DialoGPT-medium"  # Using a smaller model for quick testing

    try:
        model, tokenizer = load_model_and_tokenizer(model_name)
        print(f"Successfully loaded {model_name}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have the model available or change the model_name")
        return

    # Test prompt that requires reasoning
    prompt = "If all humans suddenly disappeared, what would happen to Earth's ecosystems over 100 years?"

    print("\n" + "="*70)
    print("COMPARISON: STANDARD GENERATION vs POWER SAMPLING")
    print("="*70)
    print(f"Prompt: {prompt}")
    print("\n" + "-"*70)

    # Standard generation
    print("\n1. STANDARD GENERATION")
    print("-" * 30)

    start_time = time.time()
    try:
        standard_response = standard_generate(model, tokenizer, prompt)
        standard_time = time.time() - start_time
        print(f"Response ({standard_time:.2f}s):\n{standard_response}")
    except Exception as e:
        print(f"Error in standard generation: {e}")
        standard_response = ""
        standard_time = 0

    print("\n" + "-"*70)

    # Power sampling
    print("\n2. POWER SAMPLING")
    print("-" * 30)

    start_time = time.time()
    try:
        power_response = power_sample(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            alpha=4.0,
            block_size=64,  # Smaller block for this model
            steps=5,        # Fewer steps for speed
            max_len=512,
            show_progress=True
        )
        power_time = time.time() - start_time
        print(f"Response ({power_time:.2f}s):\n{power_response}")
    except Exception as e:
        print(f"Error in power sampling: {e}")
        power_response = ""
        power_time = 0

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Standard Generation Time: {standard_time:.2f}s")
    print(f"Power Sampling Time: {power_time:.2f}s")
    print(f"Speed Ratio: {power_time/max(standard_time, 0.001):.1f}x")

    print("\nObservations:")
    print("- Power sampling typically produces more coherent and logically consistent responses")
    print("- Power sampling takes longer due to multiple resampling steps")
    print("- The improvement is more noticeable on complex reasoning tasks")


if __name__ == "__main__":
    main()