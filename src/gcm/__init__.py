"""Generalized contact matrices.

    from gcm import Domain, Counts, Element, ContactMatrix, heatmap
"""
from .matrix import ContactMatrix, Counts, Domain, Element
from .plot import heatmap

__all__ = ["ContactMatrix", "Counts", "Domain", "Element", "heatmap"]
