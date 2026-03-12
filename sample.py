#!/usr/bin/env python3
"""
Generate mathematical expressions by iterative denoising.

Uses stochastic sampling by default (temperature=1.0). Use --temperature 0 for
deterministic argmax.

Usage:
    python sample.py --checkpoint checkpoints/model.pt --num 10
    python sample.py --checkpoint checkpoints/model.pt --num 20 --temperature 0.9
"""

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from tokenizer import decode, PAD_ID, MASK_ID
from diffusion import NUM_TIMESTEPS
from model import DenoisingTransformer


def load_model(checkpoint_path: Path, device: torch.device):
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    max_length = ckpt.get("max_length", 50)
    model = DenoisingTransformer(max_len=max_length)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model.to(device), max_length


def sample(
    model: DenoisingTransformer,
    num_samples: int,
    seq_length: int,
    num_steps: int,
    device: torch.device,
    temperature: float = 1.0,
) -> list[str]:
    """
    Generate expressions by iterative denoising from noise.

    Starts from all [MASK]. At each step t, the model predicts original tokens.
    We reveal only a fraction of MASK positions (ceil(n_masked / t)) so that
    denoising is gradual over T steps, matching the training distribution
    where the model sees inputs with varying corruption levels.
    Positions to reveal are chosen by prediction confidence (MaskGIT-style).
    """
    # Start from all MASK
    x = torch.full((num_samples, seq_length), MASK_ID, dtype=torch.long, device=device)
    pad_mask = torch.zeros(num_samples, seq_length, dtype=torch.bool, device=device)

    for t in range(num_steps, 0, -1):
        t_tensor = torch.full((num_samples,), t, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x, t_tensor, pad_mask=pad_mask)

        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            pred = torch.multinomial(
                probs.view(-1, probs.size(-1)), 1
            ).view(logits.shape[0], logits.shape[1])
            # Confidence = probability assigned to the sampled token
            confidence = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
        else:
            pred = logits.argmax(dim=-1)
            probs = F.softmax(logits, dim=-1)
            confidence = probs.max(dim=-1).values

        mask_pos = (x == MASK_ID) & ~pad_mask
        n_masked = mask_pos.sum(dim=1)

        # Reveal ceil(n_masked / t) positions per sample, chosen by confidence
        for b in range(num_samples):
            n = n_masked[b].item()
            if n == 0:
                continue
            k = max(1, (n + t - 1) // t)  # ceil(n / t) positions to reveal

            # Get confidence for masked positions only
            conf = confidence[b].clone()
            conf = torch.where(mask_pos[b], conf, torch.full_like(conf, float("-inf")))
            _, indices = torch.topk(conf, min(k, n))

            # Replace the top-k most confident MASK positions
            for idx in indices:
                if x[b, idx] == MASK_ID:
                    x[b, idx] = pred[b, idx]

    return [decode(row.tolist()) for row in x]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate expressions via diffusion")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/model.pt"))
    parser.add_argument("--num", type=int, default=10)
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--steps", type=int, default=NUM_TIMESTEPS)
    parser.add_argument("--temperature", type=float, default=1.0,
        help="Sampling temperature (1.0=stochastic, 0=deterministic argmax)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, max_length = load_model(args.checkpoint, device)
    seq_length = args.seq_length or max_length

    exprs = sample(model, args.num, seq_length, args.steps, device,
        temperature=args.temperature)
    for e in exprs:
        print(e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
