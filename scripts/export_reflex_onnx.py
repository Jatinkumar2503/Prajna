"""
PRAJNA FAST-REFLEX ONNX & C++ EXPORT ENGINE
Converts trained 25k PrajnaFastReflex student model into optimized ONNX graph and C++ header.
"""
import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from prajna_core.models.fast_reflex import PrajnaFastReflex
