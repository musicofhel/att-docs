"""Synthetic chaotic system generators for validation."""

from att.synthetic.generators import (
    lorenz_system,
    rossler_system,
    coupled_lorenz,
    coupled_rossler_lorenz,
    switching_rossler,
    coupled_oscillators,
    kuramoto_oscillators,
    aizawa_system,
)
__all__ = [
    "lorenz_system",
    "rossler_system",
    "coupled_lorenz",
    "coupled_rossler_lorenz",
    "switching_rossler",
    "coupled_oscillators",
    "kuramoto_oscillators",
    "aizawa_system",
    "layered_aizawa_network",
]


def __getattr__(name):
    if name == "layered_aizawa_network":
        from att.synthetic.layered_network import layered_aizawa_network
        return layered_aizawa_network
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
