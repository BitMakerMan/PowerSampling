"""
Test con modello leggero per Power Sampling
"""

import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from power_sampling import load_model_and_tokenizer, power_sample


def main():
    """Test con un modello più piccolo e gestibile."""

    # Modelli più leggeri per test su CPU
    model_options = [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",      # 1.1B parametri - molto leggero
        "microsoft/DialoGPT-medium",               # 345M parametri - ancora più leggero
        "EleutherAI/gpt-neo-125M",                 # 125M parametri - ultra leggero
    ]

    print("Testing Power Sampling con modello leggero...")

    for model_name in model_options:
        print(f"\nTentativo con: {model_name}")

        try:
            print("Caricamento modello...")
            model, tokenizer = load_model_and_tokenizer(model_name)
            print(f"[OK] Modello caricato con successo: {model_name}")

            # Test semplice
            prompt = "What is artificial intelligence?"
            print(f"Prompt: {prompt}")

            response = power_sample(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                alpha=2.0,      # Ridotto per test più veloci
                block_size=64,  # Più piccolo per test
                steps=3,        # Meno iterazioni per test
                max_len=256,    # Più corto per test
                show_progress=True
            )

            print(f"Risposta: {response[:200]}...")
            print("[OK] Test completato con successo!")
            break  # Se funziona, non provare altri modelli

        except Exception as e:
            print(f"[ERROR] Errore con {model_name}: {e}")
            continue

    else:
        print("Nessun modello ha funzionato. Verifica connessione internet e dipendenze.")


if __name__ == "__main__":
    main()