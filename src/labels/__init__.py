"""Label generation modules for trading signals."""

from .signal_generation import SignalGenerator, TripleBarrierMethod
from .label_utils import create_labels, validate_labels

__all__ = ["SignalGenerator", "TripleBarrierMethod", "create_labels", "validate_labels"]
