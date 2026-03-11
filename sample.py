#!/usr/bin/env python3
"""
Generate mathematical expressions by iterative denoising.

Usage:
    python sample.py --checkpoint checkpoints/model.pt --num 10
    python sample.py --checkpoint checkpoints/model.pt --num 5 --seq-length 20
"""

import argparse
import random
import sys
from pathlib import Path

import torch

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
    rng=None,
) -> list[str]:
    """
    Generate expressions by iterative denoising from noise.

    Starts from all [MASK]. At each step, model predicts original tokens;
    we replace MASK positions with predictions and iterate.
    """
    r = rng or random
    # Start from all MASK
    x = torch.full((num_samples, seq_length), MASK_ID, dtype=torch.long, device=device)
    pad_mask = torch.zeros(num_samples, seq_length, dtype=torch.bool, device=device)

    for t in range(num_steps, 0, -1):
        t_tensor = torch.full((num_samples,), t, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x, t_tensor, pad_mask=pad_mask)
        pred = logits.argmax(dim=-1)
        # Replace MASK positions with predictions
        mask_pos = (x == MASK_ID) & ~pad_mask
        x = torch.where(mask_pos, pred, x)

    return [decode(row.tolist()) for row in x]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate expressions via diffusion")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/model.pt"))
    parser.add_argument("--num", type=int, default=10)
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--steps", type=int, default=NUM_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, max_length = load_model(args.checkpoint, device)
    seq_length = args.seq_length or max_length

    exprs = sample(model, args.num, seq_length, args.steps, device, rng=rng)
    for e in exprs:
        print(e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
