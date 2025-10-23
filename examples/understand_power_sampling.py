#!/usr/bin/env python3
"""
🧠 Understand Power Sampling - Educational Demo

This file demonstrates WHAT Power Sampling does and WHY it improves LLM reasoning.
Perfect for understanding the algorithm step by step.

Author: Educational demo by Craicek (BitMakerMan)
Based on: Original Power Sampling by aakaran
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from power_sampling import load_model_and_tokenizer, power_sample
    print("✅ Successfully imported Power Sampling modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def print_header(title):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f"🎯 {title}")
    print(f"{'='*60}")

def print_comparison(original, improved, prompt):
    """Print side-by-side comparison"""
    print(f"\n📝 PROMPT: {prompt}")
    print("-" * 60)

    print(f"🔸 STANDARD GENERATION:")
    print(f"   {original}")
    print()

    print(f"✨ POWER SAMPLING (Improved):")
    print(f"   {improved}")
    print()

    print("💡 IMPROVEMENTS:")

    # Simple analysis of improvements
    if len(improved) > len(original):
        print("   ✓ More detailed response")
    if "because" in improved.lower() or "therefore" in improved.lower():
        print("   ✓ Better logical connections")
    if "step" in improved.lower() or "first" in improved.lower():
        print("   ✓ Structured reasoning")
    if improved.count('.') > original.count('.'):
        print("   ✓ More complete sentences")

    print("-" * 60)

def demonstrate_basic_generation(model, tokenizer):
    """Show basic LLM generation without Power Sampling"""
    print_header("1. Standard LLM Generation (Without Power Sampling)")

    prompt = "What is artificial intelligence?"
    print(f"Prompt: {prompt}")
    print("This shows how a normal language model generates text...")

    # Standard generation
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=1.0,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    standard_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {standard_response}")

    return standard_response

def demonstrate_power_sampling(model, tokenizer):
    """Show Power Sampling in action"""
    print_header("2. Power Sampling in Action")

    prompt = "What is artificial intelligence?"
    print(f"Same prompt: {prompt}")
    print("Power Sampling applies Metropolis-Hastings to improve coherence...")

    # Power Sampling with different parameters
    print("\n🔧 Testing different Power Sampling parameters:")

    for alpha in [2.0, 4.0, 6.0]:
        print(f"\n--- Alpha = {alpha} (Sharpening Factor) ---")

        response = power_sample(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            alpha=alpha,
            steps=3,
            block_size=32,
            max_len=120,
            show_progress=False
        )

        print(f"Response: {response[:200]}..." if len(response) > 200 else f"Response: {response}")

def demonstrate_step_by_step(model, tokenizer):
    """Show how Power Sampling works step by step"""
    print_header("3. How Power Sampling Works - Step by Step")

    prompt = "Explain why the sky is blue"
    print(f"Prompt: {prompt}")
    print("\n🔄 Power Sampling Process:")
    print("1. Generate initial text")
    print("2. Divide text into blocks")
    print("3. Propose alternatives for each block")
    print("4. Accept/reject based on probability improvement")
    print("5. Repeat for multiple iterations")

    # Run with progress to show the steps
    print(f"\nRunning Power Sampling with visible steps...")

    response = power_sample(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        alpha=4.0,
        steps=5,
        block_size=24,
        max_len=150,
        show_progress=True  # This will show the progress
    )

    print(f"\n✅ Final Improved Response:")
    print(f"   {response}")

def compare_different_prompts(model, tokenizer):
    """Compare Power Sampling on different types of prompts"""
    print_header("4. Power Sampling on Different Question Types")

    prompts = [
        "What is machine learning?",
        "How does photosynthesis work?",
        "Why do we dream?",
        "Explain gravity simply"
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n📍 Test {i}: {prompt}")

        # Get standard response
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=60,
                temperature=1.0,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        standard = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Get Power Sampling response
        improved = power_sample(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            alpha=4.0,
            steps=3,
            block_size=20,
            max_len=100,
            show_progress=False
        )

        print_comparison(standard, improved, prompt)

def explain_theory():
    """Explain the theory behind Power Sampling"""
    print_header("5. The Theory Behind Power Sampling")

    print("""
🧠 What is Power Sampling?

Power Sampling is an algorithm that improves LLM reasoning using the
Metropolis-Hastings method (a technique from computational statistics).

🔬 How it Works:

1. INITIAL GENERATION
   - Generate text normally with the LLM
   - This gives us a starting point

2. BLOCK RESAMPLING
   - Split the text into smaller blocks
   - For each block, propose alternative text
   - Calculate probability scores for both original and new text

3. METROPOLIS-HASTINGS DECISION
   - Accept the new text IF it improves overall probability
   - Sometimes accept worse text to explore possibilities
   - This helps avoid getting stuck in local optima

4. ITERATION
   - Repeat the process multiple times
   - Each iteration typically improves the text
   - More iterations = better quality (but slower)

🎯 Why it Works:

- **Focus**: The "alpha" parameter sharpens probabilities, making better text more likely
- **Exploration**: Sometimes accepts worse options to find better solutions
- **Coherence**: Evaluates text in context, not just word by word
- **Iteration**: Multiple rounds of refinement

📊 Parameters:

- **Alpha (2.0-6.0)**: Higher = more focused on high-probability text
- **Steps (1-10)**: Number of refinement iterations
- **Block Size (16-128)**: Size of text chunks to resample
- **Max Length**: Maximum response length

💡 The Result: More coherent, logical, and well-structured text!
""")

def main():
    """Main demonstration function"""
    print_header("🧠 Understanding Power Sampling - Educational Demo")
    print("This demo will show you exactly what Power Sampling does and how it works!")
    print("Based on original work by aakaran, educational version by Craicek")

    # Load lightweight model from local directory
    # The model is stored in HuggingFace cache structure
    base_model_path = os.path.join(os.path.dirname(__file__), '..', 'models--EleutherAI--gpt-neo-125M')
    model_path = os.path.join(base_model_path, 'snapshots', '21def0189f5705e2521767faed922f1f15e7d7db')

    print(f"\n🔄 Loading model from local directory...")
    print(f"📁 Model path: {model_path}")

    # Check if model directory exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model directory not found at {model_path}")
        print("Please ensure the model is downloaded and available locally.")
        print("Alternatively, we can try to download from HuggingFace...")
        model_path = "EleutherAI/gpt-neo-125M"  # Fallback to online model
        print(f"🔄 Using online model: {model_path}")
    else:
        print(f"✅ Model directory found locally!")

    print(f"🚀 Loading model from: {model_path}")
    model, tokenizer = load_model_and_tokenizer(model_path)
    print("✅ Model loaded successfully from local directory!")

    try:
        # Explain the theory first
        explain_theory()

        # Show standard generation
        standard_response = demonstrate_basic_generation(model, tokenizer)

        # Show Power Sampling
        demonstrate_power_sampling(model, tokenizer)

        # Step by step demo
        demonstrate_step_by_step(model, tokenizer)

        # Compare different prompts
        compare_different_prompts(model, tokenizer)

        print_header("🎉 Summary")
        print("""
✅ You've seen how Power Sampling works!

Key Takeaways:
1. Power Sampling improves LLM text quality
2. Uses Metropolis-Hastings algorithm for refinement
3. Works by iteratively improving text blocks
4. Parameters control quality vs speed trade-offs
5. Results in more coherent and logical responses

📚 For more information:
- Original repository: https://github.com/aakaran/reasoning-with-sampling
- Educational demo: https://github.com/BitMakerMan/PowerSampling

🚀 Try it yourself with different prompts and parameters!
""")

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        print("This is normal - LLM generation can sometimes fail.")
        print("Try running the demo again!")

if __name__ == "__main__":
    main()