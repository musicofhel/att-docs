"""Synthetic chaotic system generators for validation."""

from att.synthetic.generators import (
    lorenz_system,
    rossler_system,
    coupled_lorenz,
    coupled_rossler_lorenz,
    switching_rossler,
    coupled_oscillators,
)

__all__ = [
    "lorenz_system",
    "rossler_system",
    "coupled_lorenz",
    "coupled_rossler_lorenz",
    "switching_rossler",
    "coupled_oscillators",
]
