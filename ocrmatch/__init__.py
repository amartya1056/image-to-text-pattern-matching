"""ocrmatch - an image-to-text pattern matching pipeline.

Stages:  input -> preprocessing -> OCR -> text cleaning -> pattern matching -> output
"""

from .config import MatchConfig, load_config
from .pipeline import Pipeline, PipelineOptions, PipelineResult
from .preprocessing import PreprocessOptions

__all__ = [
    "MatchConfig",
    "load_config",
    "Pipeline",
    "PipelineOptions",
    "PipelineResult",
    "PreprocessOptions",
]

__version__ = "1.0.0"
