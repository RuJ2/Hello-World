import sys\n\ndef hello():\n    print("hello, world")\n\ndef add(a, b):\n    return a + b\n\ndef divide(a, b):\n    return a / b\n\nif __name__ == "__main__":\n    hello()\n
def safe_divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

