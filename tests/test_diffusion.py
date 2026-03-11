import pytest
from tokenizer import encode, decode, PAD_ID, MASK_ID, encode_batch
from diffusion import gamma, corrupt, corrupt_batch, sample_timestep, NUM_TIMESTEPS


def test_gamma_bounds():
    assert gamma(0, 100) == 0.0
    assert gamma(100, 100) == 1.0


def test_gamma_monotonic():
    for t in range(1, 50):
        assert gamma(t, 50) <= gamma(t + 1, 50)


def test_corrupt_t0_unchanged():
    rng = __import__("random").Random(42)
    ids = encode("(x+2)*y")
    corrupted = corrupt(ids, 0, rng=rng)
    assert corrupted == ids


def test_corrupt_tmax_all_corrupted():
    rng = __import__("random").Random(42)
    ids = encode("xy")
    corrupted = corrupt(ids, NUM_TIMESTEPS, rng=rng)
    assert all(c == MASK_ID or c in range(2, 35) for c in corrupted)
    assert corrupted != ids


def test_corrupt_preserves_padding():
    rng = __import__("random").Random(42)
    batch = encode_batch(["x"], max_length=5)
    corrupted = corrupt_batch(batch, t=50, rng=rng)
    assert corrupted[0][1:] == [PAD_ID] * 4


def test_sample_timestep_in_range():
    rng = __import__("random").Random(42)
    for _ in range(20):
        t = sample_timestep(rng=rng)
        assert 1 <= t <= NUM_TIMESTEPS
