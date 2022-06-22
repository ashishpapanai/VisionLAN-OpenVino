from pickle import EMPTY_LIST
import cv2
import numpy as np
import torch
import pyclipper
from shapely.geometry import Polygon
import time
import openvino.runtime as ov
core = ov.Core()
model = core.read_model('VisionLAN_IR_FP32/VisionLAN.xml')
compile_model = core.compile_model(model, "MYRIAD")
infer_request = compile_model.create_infer_request()
