import pytest
from tokenizer import encode, decode, encode_batch, vocab_contains, PAD_ID, MASK_ID


@pytest.mark.parametrize("expr", ["x", "(x+2)*y", "sin(x)+3", "log(x)+sqrt(x+1)"])
def test_roundtrip(expr):
    assert decode(encode(expr)) == expr


def test_encode_batch_pads():
    batch = encode_batch(["x", "x+y"], max_length=5)
    assert len(batch[0]) == 5
    assert decode(batch[0]) == "x"


def test_encode_batch_truncates():
    batch = encode_batch(["sin(x)+cos(y)"], max_length=5)
    assert len(batch[0]) == 5
