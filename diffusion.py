"""
Discrete diffusion corruption process for mathematical expressions.

At each timestep t, characters are randomly replaced with [MASK] or random
vocabulary tokens. The noise level increases with t (t=0: clean, t=T: fully corrupted).
"""

import math
from typing import Optional

from tokenizer import PAD_ID, MASK_ID, VOCAB_SIZE, CHAR_TO_ID, ID_TO_CHAR

# Default number of diffusion timesteps
NUM_TIMESTEPS = 100

# Character tokens only (exclude [PAD] and [MASK]) for random replacement
_CHAR_IDS = [i for i in range(VOCAB_SIZE) if i != PAD_ID and i != MASK_ID]


def gamma(t: int, T: int, schedule: str = "linear") -> float:
    """
    Noise level at timestep t. Probability that a token is corrupted.

    Args:
        t: Current timestep (0 = clean, T = fully noisy).
        T: Total number of timesteps.
        schedule: "linear" or "cosine".

    Returns:
        Float in [0, 1]. At t=0 returns 0, at t=T returns ~1.
    """
    if t <= 0:
        return 0.0
    if t >= T:
        return 1.0
    s = t / T
    if schedule == "linear":
        return s
    if schedule == "cosine":
        return 1 - math.cos(0.5 * math.pi * s) ** 2
    raise ValueError(f"Unknown schedule: {schedule}")


def corrupt(
    ids: list[int],
    t: int,
    *,
    T: int = NUM_TIMESTEPS,
    schedule: str = "linear",
    mask_prob: float = 0.8,
    pad_id: int = PAD_ID,
    rng=None,
) -> list[int]:
    """
    Corrupt token IDs at timestep t.

    Non-pad positions are replaced with [MASK] (with probability mask_prob)
    or a random character token (with probability 1 - mask_prob), each
    with probability gamma(t).

    Args:
        ids: Token IDs (may be padded).
        t: Timestep (0 = no corruption, T = full corruption).
        T: Total timesteps.
        schedule: Noise schedule ("linear" or "cosine").
        mask_prob: When corrupting, use [MASK] vs random token.
        pad_id: ID to leave unchanged (padding).
        rng: Optional random.Random for reproducibility.

    Returns:
        Corrupted copy of ids.
    """
    import random
    r = rng or random

    g = gamma(t, T, schedule)
    result = list(ids)

    for i in range(len(result)):
        if result[i] == pad_id:
            continue
        if r.random() < g:
            if r.random() < mask_prob:
                result[i] = MASK_ID
            else:
                result[i] = r.choice(_CHAR_IDS)

    return result


def corrupt_batch(
    batch: list[list[int]],
    t: int,
    *,
    T: int = NUM_TIMESTEPS,
    schedule: str = "linear",
    mask_prob: float = 0.8,
    pad_id: int = PAD_ID,
    rng=None,
) -> list[list[int]]:
    """Corrupt a batch of token ID sequences."""
    return [corrupt(ids, t, T=T, schedule=schedule, mask_prob=mask_prob, pad_id=pad_id, rng=rng) for ids in batch]


def sample_timestep(T: int = NUM_TIMESTEPS, rng=None) -> int:
    """Sample a random timestep in [1, T] for training."""
    import random
    r = rng or random
    return r.randint(1, T)
