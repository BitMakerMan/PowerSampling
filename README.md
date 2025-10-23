# Reasoning with Power Sampling

Una libreria Python per migliorare le capacità di ragionamento e coerenza logica dei modelli linguistici (LLM) attraverso l'algoritmo **Power Sampling** (Metropolis-Hastings Autoregressivo).

## 🎯 Obiettivo

Power Sampling utilizza l'algoritmo di Metropolis-Hastings per ricampionare blocchi di testo, migliorando significativamente:
- Coerenza logica nelle risposte
- Capacità di ragionamento complesso
- Consistenza nelle argomentazioni
- Qualità delle spiegazioni scientifiche e matematiche

## 📋 Requisiti

- Python >= 3.10
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- CUDA raccomandato (ma non obbligatorio)

## 🚀 Installazione Rapida

### 1. Clona il Repository

```bash
git clone https://github.com/aakaran/reasoning-with-sampling.git
cd reasoning-with-sampling
```

### 2. Installa le Dipendenze

```bash
# Opzione A: Installazione completa
pip install -r requirements.txt

# Opzione B: Dipendenze principali
pip install torch transformers numpy tqdm accelerate
```

## 📖 Utilizzo Base

### Caricamento del Modello

```python
from src.power_sampling import load_model_and_tokenizer, power_sample

# Carica modello e tokenizer
model_name = "Qwen/Qwen2.5-7B-Instruct"  # o "microsoft/DialoGPT-medium"
model, tokenizer = load_model_and_tokenizer(model_name)
```

### Power Sampling Semplice

```python
# Prompt per ragionamento complesso
prompt = "Explain how quantum entanglement works in simple terms."

# Esegui Power Sampling
response = power_sample(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    alpha=4.0,      # Fattore di sharpening
    block_size=192, # Dimensione blocco
    steps=10,       # Iterazioni MCMC
    max_len=2048,   # Lunghezza massima
    show_progress=True
)

print(response)
```

### Utilizzo Avanzato con la Classe PowerSampler

```python
from src.power_sampling import PowerSampler

# Inizializza il sampler
sampler = PowerSampler(model, tokenizer)

# Esegui sampling con parametri personalizzati
response = sampler.power_sample(
    prompt="What are the main causes of climate change?",
    alpha=4.0,
    block_size=192,
    steps=10,
    temperature=0.9,
    show_progress=True
)
```

### Batch Processing

```python
# Processa multiple prompt in batch
prompts = [
    "How do neural networks learn?",
    "What causes earthquakes?",
    "Why is the sky blue?"
]

results = sampler.batch_power_sample(
    prompts,
    alpha=3.0,
    block_size=128,
    steps=5,
    show_progress=True
)

for prompt, result in zip(prompts, results):
    print(f"Q: {prompt}")
    print(f"A: {result}\n")
```

## ⚙️ Parametri di Ottimizzazione

| Parametro | Descrizione | Valore Consigliato | Note |
|-----------|-------------|-------------------|------|
| `alpha` | Fattore di sharpening | **4.0** | Controlla l'accentuazione del campionamento |
| `steps` | Iterazioni Metropolis-Hastings | **10** | Maggiore = migliore ragionamento, ma più lento |
| `block_size` | Dimensione blocco ricampionamento | **192** | Circa T/16 dove T è la lunghezza target |
| `temperature` | Temperatura di sampling | **1.0** | Valori più bassi = più deterministico |
| `max_len` | Lunghezza massima sequenza | **2048** | Dipende dal modello |

## 🧪 Esempi Pratici

### 1. Spiegazioni Scientifiche

```python
prompt = "Explain how photosynthesis works and why it's important for life on Earth."
response = power_sample(model, tokenizer, prompt, alpha=4.0, steps=10)
```

### 2. Risoluzione di Problemi

```python
prompt = "If a train travels 300km in 3 hours, and another train travels 450km in 4.5 hours, which train is faster and by how much?"
response = power_sample(model, tokenizer, prompt, alpha=4.0, steps=15)
```

### 3. Ragionamento Causale

```python
prompt = "What would happen to Earth's ecosystems if humans suddenly disappeared? Explain the cascade effects over 100 years."
response = power_sample(model, tokenizer, prompt, alpha=4.0, steps=12)
```

## 📊 Performance vs Generazione Standard

| Caratteristica | Generazione Standard | Power Sampling |
|----------------|---------------------|----------------|
| Velocità | Veloce | Più lento (2-10x) |
| Coerenza | Media | Alta |
| Ragionamento | Basico | Avanzato |
| Consistenza | Variabile | Stabile |

## 🛠️ Script di Esempio

Il progetto include diversi script di esempio:

- **`examples/basic_usage.py`**: Utilizzo base di Power Sampling
- **`examples/comparison.py`**: Confronto tra generazione standard e Power Sampling
- **`examples/batch_processing.py`**: Processamento batch di prompt

### Esegui gli Esempi

```bash
# Esempio base
python examples/basic_usage.py

# Confronto performance
python examples/comparison.py

# Batch processing
python examples/batch_processing.py
```

## 🔧 Troubleshooting

### Problemi Comuni

1. **CUDA Out of Memory**
   ```python
   # Usa un modello più piccolo
   model_name = "microsoft/DialoGPT-small"

   # Riduci max_len e block_size
   response = power_sample(model, tokenizer, prompt, max_len=512, block_size=64)
   ```

2. **Modello Non Trovato**
   ```bash
   # Installa dipendenze aggiuntive
   pip install accelerate
   ```

3. **Risultati Lenti**
   ```python
   # Riduci il numero di steps
   response = power_sample(model, tokenizer, prompt, steps=5)

   # Usa block_size più piccolo
   response = power_sample(model, tokenizer, prompt, block_size=128)
   ```

## 🤝 Contributi

Contributi sono benvenuti! Per favore:

1. Fai un fork del repository
2. Crea una branch per la tua feature (`git checkout -b feature/amazing-feature`)
3. Fai commit delle tue modifiche (`git commit -m 'Add amazing feature'`)
4. Fai push alla branch (`git push origin feature/amazing-feature`)
5. Apri una Pull Request

## 📄 Licenza

Questo progetto è distribuito sotto licenza MIT - vedi il file [LICENSE](LICENSE) per dettagli.

## 🙏 Riconoscimenti

- Basato sull'algoritmo Power Sampling (Metropolis-Hastings Autoregressivo)
- Ispirato da "Reasoning with Sampling" paper
- Supportato dalla comunità Hugging Face 🤗

## 📞 Contatti

Per domande o supporto:

- Apri una issue su GitHub
- Contatta il team di sviluppo

---

**Nota**: Power Sampling è particolarmente efficace per task che richiedono ragionamento complesso, spiegazioni dettagliate e argomentazioni logiche. Per task semplici, la generazione standard potrebbe essere sufficiente.
