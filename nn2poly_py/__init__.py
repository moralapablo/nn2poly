"""
nn2poly_py: Convert Neural Networks to Polynomials

This package provides tools to transform trained neural networks into
their equivalent polynomial representations. This can be useful for
analysis, interpretation, and potentially for more efficient evaluation
in certain contexts.
"""

# Import the main function for easier access
from .core import nn2poly

# Import model parsers for easier access
from .model_parsers import parse_keras_model, parse_pytorch_model

# Define __all__ for explicit public API if desired (optional)
__all__ = ['nn2poly', 'parse_keras_model', 'parse_pytorch_model']

# You can also set a package-level version
__version__ = '0.1.0'
