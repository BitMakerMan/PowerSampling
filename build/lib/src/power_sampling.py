"""
Power Sampling Implementation for Reasoning with LLMs

This module implements the Power Sampling algorithm (Metropolis-Hastings Autoregressive)
to improve logical consistency and reasoning capabilities of language models.
"""

import torch
import torch.nn.functional as F
import random
import numpy as np
from typing import Optional, Dict, Any
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class PowerSampler:
    """
    Implements Power Sampling algorithm for improved LLM reasoning.

    Power Sampling uses Metropolis-Hastings algorithm to resample text blocks,
    improving logical coherence and reasoning capabilities.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: Optional[str] = None
    ):
        """
        Initialize the Power Sampler.

        Args:
            model: The causal language model to use
            tokenizer: The tokenizer corresponding to the model
            device: Device to run on (auto-detect if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def compute_log_probability(self, sequence: torch.Tensor) -> float:
        """
        Compute the log probability of a sequence under the model.

        Args:
            sequence: Token sequence tensor

        Returns:
            Log probability sum
        """
        with torch.no_grad():
            inputs = sequence.unsqueeze(0) if sequence.dim() == 1 else sequence
            attention_mask = torch.ones_like(inputs)

            outputs = self.model(
                input_ids=inputs,
                attention_mask=attention_mask,
                labels=inputs
            )

            return -outputs.loss.item() * inputs.size(-1)

    def power_sample(
        self,
        prompt: str,
        alpha: float = 4.0,
        block_size: int = 192,
        steps: int = 10,
        max_len: int = 2048,
        temperature: float = 1.0,
        show_progress: bool = False
    ) -> str:
        """
        Execute Power Sampling algorithm.

        Args:
            prompt: Input prompt text
            alpha: Sharpening factor (recommended: 4.0)
            block_size: Size of text blocks to resample (recommended: T/16)
            steps: Number of Metropolis-Hastings iterations (recommended: 10)
            max_len: Maximum sequence length
            temperature: Sampling temperature
            show_progress: Show progress bar

        Returns:
            Generated text with improved reasoning
        """
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_len - block_size
        ).to(self.device)

        # Generate initial sequence
        with torch.no_grad():
            initial_output = self.model.generate(
                **inputs,
                max_new_tokens=block_size,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )

        # Extract generated sequence (remove prompt part)
        prompt_length = inputs['input_ids'].size(-1)
        current_seq = initial_output[0][prompt_length:]

        # Metropolis-Hastings iterations
        iterator = tqdm(range(steps), desc="Power Sampling") if show_progress else range(steps)

        for step in iterator:
            # Select random block to resample
            if current_seq.size(-1) <= block_size:
                # If sequence is shorter than block_size, regenerate whole thing
                start_pos = 0
                end_pos = current_seq.size(-1)
            else:
                start_pos = random.randint(0, current_seq.size(-1) - block_size)
                end_pos = start_pos + block_size

            # Create context (prefix)
            if start_pos > 0:
                context_ids = current_seq[:start_pos].unsqueeze(0)
            else:
                # Use original prompt as context if we're at the beginning
                context_ids = inputs['input_ids']

            # Generate proposal block
            with torch.no_grad():
                proposal_block = self.model.generate(
                    context_ids,
                    max_new_tokens=block_size,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )

            # Extract only the newly generated part
            if start_pos == 0:
                new_tokens = proposal_block[0][context_ids.size(-1):]
            else:
                new_tokens = proposal_block[0][context_ids.size(-1):]

            # Create proposal sequence
            proposal_seq = torch.cat([
                current_seq[:start_pos],
                new_tokens[:block_size],
                current_seq[end_pos:] if end_pos < current_seq.size(-1) else torch.tensor([], device=self.device)
            ])

            # Ensure we don't exceed max length
            if proposal_seq.size(-1) > max_len - prompt_length:
                proposal_seq = proposal_seq[:max_len - prompt_length]
                current_seq = current_seq[:max_len - prompt_length]

            # Compute acceptance probability
            try:
                log_p_current = self.compute_log_probability(current_seq)
                log_p_proposal = self.compute_log_probability(proposal_seq)

                # Metropolis acceptance criterion with sharpening
                log_acceptance_ratio = alpha * (log_p_proposal - log_p_current)
                acceptance_prob = min(1.0, torch.exp(torch.tensor(log_acceptance_ratio)).item())

                # Accept or reject proposal
                if random.random() < acceptance_prob:
                    current_seq = proposal_seq
                    if show_progress:
                        iterator.set_postfix({"accept": f"{acceptance_prob:.3f}", "status": "✓"})
                else:
                    if show_progress:
                        iterator.set_postfix({"accept": f"{acceptance_prob:.3f}", "status": "✗"})

            except Exception as e:
                # If probability computation fails, keep current sequence
                if show_progress:
                    iterator.set_postfix({"error": str(e)[:20]})
                continue

        # Combine prompt and generated sequence
        final_sequence = torch.cat([inputs['input_ids'][0], current_seq])

        # Decode and return
        return self.tokenizer.decode(final_sequence, skip_special_tokens=True)

    def batch_power_sample(
        self,
        prompts: list,
        alpha: float = 4.0,
        block_size: int = 192,
        steps: int = 10,
        max_len: int = 2048,
        temperature: float = 1.0,
        show_progress: bool = False
    ) -> list:
        """
        Execute Power Sampling on multiple prompts.

        Args:
            prompts: List of input prompts
            alpha: Sharpening factor
            block_size: Size of text blocks to resample
            steps: Number of Metropolis-Hastings iterations
            max_len: Maximum sequence length
            temperature: Sampling temperature
            show_progress: Show progress bar

        Returns:
            List of generated texts
        """
        results = []
        iterator = tqdm(prompts, desc="Batch Power Sampling") if show_progress else prompts

        for prompt in iterator:
            result = self.power_sample(
                prompt=prompt,
                alpha=alpha,
                block_size=block_size,
                steps=steps,
                max_len=max_len,
                temperature=temperature,
                show_progress=False
            )
            results.append(result)

        return results


def load_model_and_tokenizer(model_name: str, device: Optional[str] = None):
    """
    Convenience function to load model and tokenizer.

    Args:
        model_name: Hugging Face model name
        device: Device to load on

    Returns:
        Tuple of (model, tokenizer)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )

    if device != "cuda":
        model = model.to(device)

    return model, tokenizer


def power_sample(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    alpha: float = 4.0,
    block_size: int = 192,
    steps: int = 10,
    max_len: int = 2048,
    temperature: float = 1.0,
    device: Optional[str] = None,
    show_progress: bool = False
) -> str:
    """
    Convenience function for single-call power sampling.

    Args:
        model: The causal language model
        tokenizer: The tokenizer
        prompt: Input prompt text
        alpha: Sharpening factor (recommended: 4.0)
        block_size: Size of text blocks to resample
        steps: Number of Metropolis-Hastings iterations
        max_len: Maximum sequence length
        temperature: Sampling temperature
        device: Device to run on
        show_progress: Show progress bar

    Returns:
        Generated text with improved reasoning
    """
    sampler = PowerSampler(model, tokenizer, device)
    return sampler.power_sample(
        prompt=prompt,
        alpha=alpha,
        block_size=block_size,
        steps=steps,
        max_len=max_len,
        temperature=temperature,
        show_progress=show_progress
    )