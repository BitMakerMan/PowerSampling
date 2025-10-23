"""
Reasoning with Power Sampling

A Python implementation of Power Sampling algorithm for improving
logical reasoning capabilities of language models.
"""

from .power_sampling import (
    PowerSampler,
    power_sample,
    load_model_and_tokenizer
)

__version__ = "1.0.0"
__author__ = "Power Sampling Team"

__all__ = [
    "PowerSampler",
    "power_sample",
    "load_model_and_tokenizer"
]