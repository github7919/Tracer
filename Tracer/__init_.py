# Tracer/__init__.py

"""
Tracer - A Python library for visualizing and analyzing the structure,
memory usage, and relationships of Python objects.

Modules:
- tree: Functions for visualizing the structure of nested objects.
- memory: Functions for analyzing memory usage of objects.
- reference: Functions for tracking object references and relationships.
- visualization: Functions for creating graphical representations of objects.
- utils: Utility functions for internal use.
"""

# Importing core functionalities to expose them at the package level
from .tree import visualize_tree
from .memory import memory_usage
from .reference import reference_tracking
from .visualization import visualize_relationships

# Metadata
__version__ = "0.1.0"
__author__ = "Ardhendu"
__email__ = "githubapps7919@gmail.com"

__all__ = [
    "visualize_tree",
    "memory_usage",
    "reference_tracking",
    "visualize_relationships",
]
