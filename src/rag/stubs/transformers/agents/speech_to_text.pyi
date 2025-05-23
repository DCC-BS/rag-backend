"""
This type stub file was generated by pyright.
"""

from ..models.whisper import WhisperForConditionalGeneration, WhisperProcessor
from .tools import PipelineTool

class SpeechToTextTool(PipelineTool):
    default_checkpoint = ...
    description = ...
    name = ...
    pre_processor_class = WhisperProcessor
    model_class = WhisperForConditionalGeneration
    inputs = ...
    output_type = ...
    def encode(self, audio): ...
    def forward(self, inputs): ...
    def decode(self, outputs): ...
