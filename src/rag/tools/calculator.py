from langchain_core.tools import tool
from langgraph.config import get_stream_writer


@tool("multiply", description="Multiplies two numbers")
def multiply(a: float, b: float) -> float:
    """
    Multiplies two numbers.

    Args:
        a: The first number
        b: The second number

    Returns:
        The product of the two numbers
    """
    writer = get_stream_writer()
    writer(f"Multipliziere {a} mit {b}")
    return a * b


@tool("add", description="Adds two numbers")
def add(a: float, b: float) -> float:
    """
    Adds two numbers.

    Args:
        a: The first number
        b: The second number

    Returns:
        The sum of the two numbers
    """
    writer = get_stream_writer()
    writer(f"Addiere {a} und {b}")
    return a + b


@tool("subtract", description="Subtracts two numbers")
def subtract(a: float, b: float) -> float:
    """
    Subtracts two numbers.

    Args:
        a: The first number
        b: The second number

    Returns:
        The difference of the two numbers
    """
    writer = get_stream_writer()
    writer(f"Subtrahiere {a} von {b}")
    return a - b


@tool("divide", description="Divides two numbers")
def divide(a: float, b: float) -> float:
    """
    Divides two numbers.

    Args:
        a: The first number
        b: The second number

    Returns:
        The quotient of the two numbers
    """
    writer = get_stream_writer()
    writer(f"Dividiere {a} durch {b}")
    return a / b


@tool("square", description="Squares a number")
def square(a: float) -> float:
    """
    Squares a number.

    Args:
        a: The number to square

    Returns:
        The square of the number
    """
    writer = get_stream_writer()
    writer(f"Quadriere {a}")
    return a * a
