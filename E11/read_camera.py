import numpy as np
import ctypes

_lib = np.ctypeslib.load_library("libmycamera", ".")

_lib.read_camera.argtypes = [ctypes.c_char_p]
_lib.read_camera.restype = ctypes.c_int

def read_camera():
    buffer = ctypes.create_string_buffer(1536 * 1024 * 2)
    _lib.read_camera(buffer)
    return buffer