"""
LLM Training Lab - A comprehensive toolkit for LLM dataset preparation and training.
"""

__version__ = "1.0.0"
__author__ = "LLM Training Lab Team"
__email__ = "contact@llm-training-lab.com"

from .pipeline import DataPipeline
from .config import Config

__all__ = ["DataPipeline", "Config"]