# model/__init__.py
from .model_drone import DroneMultiClassifier
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "leaf_pytorch"))

__all__ = ["DroneMultiClassifier"]
