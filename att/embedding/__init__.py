"""Phase-space reconstruction via time-delay embedding."""

from att.embedding.takens import TakensEmbedder
from att.embedding.joint import JointEmbedder
from att.embedding.delay import estimate_delay
from att.embedding.dimension import estimate_dimension
from att.embedding.validation import (
    validate_embedding,
    svd_embedding,
    EmbeddingDegeneracyWarning,
)

__all__ = [
    "TakensEmbedder",
    "JointEmbedder",
    "estimate_delay",
    "estimate_dimension",
    "validate_embedding",
    "svd_embedding",
    "EmbeddingDegeneracyWarning",
]
