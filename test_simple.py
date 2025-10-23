"""
Test semplice che funziona ovunque
"""

import sys
import os

# Aggiungi la directory src al path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

# Import che funziona sempre
import power_sampling

def main():
    print("=== Test Power Sampling ===")

    # Carica modello
    model_name = "EleutherAI/gpt-neo-125M"
    print(f"Loading model: {model_name}")

    model, tokenizer = power_sampling.load_model_and_tokenizer(model_name)
    print("[OK] Model loaded!")

    # Test Power Sampling
    prompt = "What is artificial intelligence?"
    # Prova con parametri diversi per vedere differenze
    print("\n--- Test 1: Veloce (2 steps) ---")
    response1 = power_sampling.power_sample(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        alpha=2.0,      # Meno sharpening
        steps=2,        # Veloce
        show_progress=True
    )
    print(f"Risposta veloce: {response1[:100]}...")

    print("\n--- Test 2: Qualità (5 steps) ---")
    response2 = power_sampling.power_sample(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        alpha=4.0,      # Più sharpening
        steps=5,        # Più qualità
        show_progress=True
    )
    print(f"Risposta qualità: {response2[:100]}...")

    print("\n--- Test 3: Focus (alpha=6) ---")
    response3 = power_sampling.power_sample(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        alpha=6.0,      # Massimo sharpening
        steps=3,
        show_progress=True
    )
    print(f"Risposta focus: {response3[:100]}...")

    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")

if __name__ == "__main__":
    main()