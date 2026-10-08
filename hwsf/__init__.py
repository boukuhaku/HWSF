"""HWSF: Hierarchical Weather-SST Fusion for preharvest municipal rice yield estimation."""

__version__ = "1.0.0"

from .data import TargetYearInputs, load_inputs, save_inputs  # noqa: F401
from .model import HWSF  # noqa: F401
