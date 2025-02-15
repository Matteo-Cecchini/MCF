import numpy as np
import ctypes
import tensorflow

_lib = np.ctypeslib.load_library("libmycamera", ".")

_lib.read_camera.argtypes = [ctypes.c_char_p]
_lib.read_camera.restype = ctypes.c_int

class myCamera:
    imag = np.array([])
    
    def read_camera(self):
        buffer = ctypes.create_string_buffer()