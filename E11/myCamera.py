import numpy as np
import ctypes

_lib = np.ctypeslib.load_library("libmycamera", ".")

_lib.read_camera.argtypes = [ctypes.c_char_p]
_lib.read_camera.restype = ctypes.c_int

class myCamera:
    imag = np.array()