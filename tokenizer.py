"""
Character-level tokenizer for mathematical expressions.

Vocabulary includes all characters from the grammar (grammar.py) plus special tokens
for diffusion: [PAD] and [MASK].
"""

from grammar import VARIABLES, FUNCTIONS, DIGITS

# Characters used by the grammar
OPERATORS = "+-*/^"
PARENS = "()"

# All characters that can appear in valid expressions
_CHARS = set(DIGITS) | set(OPERATORS) | set(PARENS) | set(VARIABLES)
for f in FUNCTIONS:
    _CHARS.update(f)

# Special tokens for diffusion
PAD_TOKEN = "[PAD]"
MASK_TOKEN = "[MASK]"

# Build vocabulary: special tokens first, then sorted characters
SPECIAL_TOKENS = [PAD_TOKEN, MASK_TOKEN]
CHARS = sorted(_CHARS)
VOCAB = SPECIAL_TOKENS + CHARS

# Mappings
CHAR_TO_ID = {c: i for i, c in enumerate(VOCAB)}
ID_TO_CHAR = {i: c for i, c in enumerate(VOCAB)}

PAD_ID = CHAR_TO_ID[PAD_TOKEN]
MASK_ID = CHAR_TO_ID[MASK_TOKEN]

VOCAB_SIZE = len(VOCAB)


def encode(expr: str) -> list[int]:
    """Encode an expression string to a list of token IDs."""
    return [CHAR_TO_ID[c] for c in expr]


def decode(ids: list[int], skip_pad: bool = True) -> str:
    """Decode a list of token IDs back to a string."""
    chars = []
    for i in ids:
        c = ID_TO_CHAR.get(i, "?")
        if skip_pad and c == PAD_TOKEN:
            continue
        chars.append(c)
    return "".join(chars)


def encode_batch(
    exprs: list[str],
    max_length: int,
    pad_id: int = PAD_ID,
) -> list[list[int]]:
    """
    Encode a batch of expressions to fixed-length sequences.

    Shorter sequences are right-padded with pad_id.
    Longer sequences are truncated from the right.
    """
    result = []
    for expr in exprs:
        ids = encode(expr)
        if len(ids) > max_length:
            ids = ids[:max_length]
        elif len(ids) < max_length:
            ids = ids + [pad_id] * (max_length - len(ids))
        result.append(ids)
    return result


def vocab_contains(expr: str) -> bool:
    """Check if all characters in expr are in the vocabulary."""
    return all(c in CHAR_TO_ID for c in expr)
