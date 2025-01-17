"""
VILA Vision Language Model integration for omOS.

This module provides integration with the VILA (Vision Language) model
for real-time video analysis and description generation.
"""

from .vila_processor import VILAProcessor
from .args import VILAArgParser

__all__ = ["VILAProcessor", "VILAArgParser"]
