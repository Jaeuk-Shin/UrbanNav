"""
OT-CFM velocity field.

The network architecture is identical to the standard CFM model — OT-CFM only
changes the *training procedure* (mini-batch optimal-transport pairing of
source and target samples), not the velocity-field parameterisation.

This module re-exports ``FlowMatchingMLP`` from ``model.py`` so that
``train_otcfm.py`` has a self-consistent import.
"""

from model import FlowMatchingMLP  # noqa: F401

__all__ = ["FlowMatchingMLP"]
