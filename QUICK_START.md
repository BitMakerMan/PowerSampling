# Power Sampling - Quick Start Guide

## 🚀 Quick Usage

### 1. Command Line (Fastest)
```bash
power-sampling
```

### 2. Python - Basic Usage
```python
from src.power_sampling import load_model_and_tokenizer, power_sample

# Load model
model, tokenizer = load_model_and_tokenizer("EleutherAI/gpt-neo-125M")

# Generate improved response
response = power_sample(
    model=model,
    tokenizer=tokenizer,
    prompt="What is artificial intelligence?",
    alpha=4.0,          # Sharpening factor (2-6 recommended)
    block_size=64,      # Block size for resampling
    steps=5,            # Number of iterations (1-10 recommended)
    show_progress=True
)
print(response)
```

### 3. Python - Advanced Usage
```python
from src.power_sampling import PowerSampler

# Create sampler instance
sampler = PowerSampler(model, tokenizer)

# Use multiple prompts
prompts = ["What is AI?", "How does ML work?"]
results = sampler.batch_power_sample(prompts, steps=3)
```

## 📋 Key Parameters

| Parameter | Recommended Range | Description |
|-----------|------------------|-------------|
| `alpha` | 2.0 - 6.0 | Sharpening factor (higher = more focused) |
| `block_size` | 32 - 128 | Text block size for resampling |
| `steps` | 1 - 10 | Number of Metropolis-Hastings iterations |
| `max_len` | 128 - 512 | Maximum response length |

## 🎯 Best Results

- **Quality vs Speed**: More steps = better quality but slower
- **Model Size**: Larger models give better results but need more RAM
- **Alpha**: Start with 4.0, adjust based on desired focus

## 📁 Example Files

- `examples/test_final.py` - Quick test
- `examples/usage_guide.py` - Complete examples
- `examples/basic_usage.py` - Original example with large model

## 🔧 Installation

```bash
# Install package
python setup.py install

# Test installation
power-sampling
```

## ⚠️ Requirements

- Python 3.8+
- PyTorch
- Transformers
- 4GB+ RAM for small models
- Internet connection for first model download