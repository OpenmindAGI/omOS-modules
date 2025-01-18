import argparse
from typing import Dict, List, Type, TypeVar

from ..vila_vlm.args import VILAArgParser
from ..nv_nano_llm.args import NanoLLMArgParser

# VLM Processors
from ..nv_nano_llm import VLMProcessor as NanoLLMProcessor
from ..vila_vlm import VILAProcessor as VILAProcessor

# Video device input
from ..nv_nano_llm import VideoDeviceInput as NanoLLMVideoDeviceInput

# Video stream input
from ..nv_nano_llm import VideoStreamInput as NanoLLMVideoStreamInput

T_Parser = TypeVar('T_Parser', bound=argparse.ArgumentParser)
T_Processor = TypeVar('T_Processor', NanoLLMProcessor, VILAProcessor)
T_VideoInput = TypeVar('T_VideoInput', NanoLLMVideoDeviceInput, NanoLLMVideoStreamInput)

MODEL_CONFIGS: Dict[str, List[str]] = {
    'vila': ['vila', 'standalone'],
    'nano_llm': ['model', 'chat', 'generation'],
}

MODEL_PARSERS: Dict[str, Type[T_Parser]] = {
    'vila': VILAArgParser,
    'nano_llm': NanoLLMArgParser,
}

VLM_PROCESSOR: Dict[str, Type[T_Processor]] = {
    'vila': VILAProcessor,
    'nano_llm': NanoLLMProcessor,
}

VIDEO_DEVICE_INPUT: Dict[str, Type[T_VideoInput]] = {
    'nano_llm': NanoLLMVideoDeviceInput,
}

VIDEO_STREAM_INPUT: Dict[str, Type[T_VideoInput]] = {
    'nano_llm': NanoLLMVideoStreamInput,
}