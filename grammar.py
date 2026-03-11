"""
Formal grammar for mathematical expressions.

Produces valid expressions such as:
    (x+2)*y
    sin(x)+3
    (x^2+1)/(y-4)
    log(x)+sqrt(x+1)

Grammar (operator precedence: +,- < *,/ < ^):
    expr   := term (('+' | '-') term)*
    term   := factor (('*' | '/') factor)*
    factor := base ('^' factor)?
    base   := number | variable | '(' expr ')' | func '(' expr ')'
    number := digit+
    digit  := '0'..'9'
    variable := 'x' | 'y' | 'z'
    func   := 'sin' | 'cos' | 'tan' | 'log' | 'sqrt' | 'exp'
"""

import random
from typing import Optional


# Grammar constants
VARIABLES = ['x', 'y', 'z']
FUNCTIONS = ['sin', 'cos', 'tan', 'log', 'sqrt', 'exp']
DIGITS = '0123456789'


def _random_digit() -> str:
    return random.choice(DIGITS)


def _random_number(max_digits: int = 3) -> str:
    """Generate a random integer, avoiding leading zeros."""
    n_digits = random.randint(1, max_digits)
    first = random.choice('123456789') if n_digits > 1 else random.choice(DIGITS)
    rest = ''.join(_random_digit() for _ in range(n_digits - 1))
    return first + rest if n_digits > 1 else first


def _random_variable() -> str:
    return random.choice(VARIABLES)


def _random_func() -> str:
    return random.choice(FUNCTIONS)


def generate_expr(
    max_depth: int = 3,
    max_add_terms: int = 2,
    max_mul_factors: int = 2,
    number_max_digits: int = 1,
    max_length: Optional[int] = 50,
    rng: Optional[random.Random] = None,
) -> str:
    """
    Generate a random valid mathematical expression.

    Args:
        max_depth: Maximum nesting depth (parentheses, function args).
        max_add_terms: Max number of +/- terms at top level (1–N).
        max_mul_factors: Max number of *// factors per term (1–N).
        number_max_digits: Max digits in generated numbers.
        max_length: If set, no expression may exceed this length (retries until satisfied).
        rng: Optional random.Random instance for reproducibility.

    Returns:
        A valid mathematical expression string.
    """
    r = rng or random

    def expr(depth: int) -> str:
        if depth <= 0:
            return base(depth)
        n_terms = r.randint(1, max_add_terms)
        parts = [term(depth) for _ in range(n_terms)]
        ops = r.choices(['+', '-'], k=n_terms - 1)
        result = parts[0]
        for op, part in zip(ops, parts[1:]):
            result += op + part
        return result

    def term(depth: int) -> str:
        if depth <= 0:
            return factor(depth)
        n_factors = r.randint(1, max_mul_factors)
        parts = [factor(depth) for _ in range(n_factors)]
        ops = r.choices(['*', '/'], k=n_factors - 1)
        result = parts[0]
        for op, part in zip(ops, parts[1:]):
            result += op + part
        return result

    def factor(depth: int) -> str:
        b = base(depth)
        if depth > 0 and r.random() < 0.2:
            return b + '^' + factor(depth - 1)
        return b

    def base(depth: int) -> str:
        if depth <= 0:
            return r.choice([_random_number(number_max_digits), _random_variable()])
        choices = [
            lambda: _random_number(number_max_digits),
            lambda: _random_variable(),
        ]
        if depth >= 1:
            choices.append(lambda: '(' + expr(depth - 1) + ')')
        if depth >= 2:
            choices.append(lambda: _random_func() + '(' + expr(depth - 1) + ')')
        return r.choice(choices)()

    for _ in range(500):
        result = expr(max_depth)
        if max_length is None or len(result) <= max_length:
            return result
    return expr(max_depth)


def generate_dataset(
    n: int,
    *,
    max_depth: int = 3,
    max_add_terms: int = 2,
    max_mul_factors: int = 2,
    number_max_digits: int = 1,
    max_length: Optional[int] = 50,
    seed: Optional[int] = None,
    deduplicate: bool = True,
) -> list[str]:
    """
    Generate a dataset of n unique mathematical expressions.

    Args:
        n: Number of expressions to generate.
        max_depth: Maximum nesting depth.
        max_add_terms: Max +/- terms per expression.
        max_mul_factors: Max *// factors per term.
        number_max_digits: Max digits in numbers.
        max_length: If set, no expression may exceed this length.
        seed: Random seed for reproducibility.
        deduplicate: If True, ensure all expressions are unique.

    Returns:
        List of expression strings.
    """
    rng = random.Random(seed)
    seen: set[str] = set()
    result: list[str] = []

    while len(result) < n:
        expr_str = generate_expr(
            max_depth=max_depth,
            max_add_terms=max_add_terms,
            max_mul_factors=max_mul_factors,
            number_max_digits=number_max_digits,
            max_length=max_length,
            rng=rng,
        )
        if not deduplicate or expr_str not in seen:
            seen.add(expr_str)
            result.append(expr_str)

    return result
