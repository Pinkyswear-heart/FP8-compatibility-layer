"""TFP8: Turing FP8-compatible runtime (W8A16 demo).

FP8 weights are stored as uint8 codes + f32 scales, decoded to FP16 inside the tile and run on
FP16 tensor cores with FP32 accumulation. It saves memory and bandwidth; it does not fake
native FP8 throughput.
"""
from .config import FP8Config
from .formats import E4M3FN, E5M2, decode, encode, lut
from .linear import FP8Linear
from .planner import plan, prepare_model, report, weight_bytes
from .quantize import dequantize_weight, quantize_weight

__all__ = ["FP8Config", "FP8Linear", "prepare_model", "plan", "report", "weight_bytes",
           "quantize_weight", "dequantize_weight", "decode", "encode", "lut", "E4M3FN", "E5M2"]
