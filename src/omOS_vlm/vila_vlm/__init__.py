"""
VILA Vision Language Model integration for omOS.

This module provides integration with the VILA (Vision Language) model
for real-time video analysis and description generation.
"""

from .vila_processor import VILAProcessor
from .args import VILAArgParser
from .__main__ import main

__all__ = ["VILAProcessor", "VILAArgParser", "main"]
