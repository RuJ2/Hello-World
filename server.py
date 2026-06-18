"""Hello World server with utility functions."""

def hello():
    """Print a greeting."""
    print("hello, world")


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def divide(a: float, b: float) -> float:
    """Divide a by b. Beware: no zero check!"""
    return a / b


def safe_divide(a: float, b: float) -> float:
    """Divide a by b with zero check."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


if __name__ == "__main__":
    hello()
