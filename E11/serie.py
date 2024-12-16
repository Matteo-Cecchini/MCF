import ctypes as ct
import numpy as np

_libserie = np.ctypeslib.load_library("libserie", ".")

_libserie.fibonacci.argtypes = [ct.c_int]
_libserie.fibonacci.restype = ct.c_longdouble

def fibonacci(n: int):
    return _libserie.fibonacci(int(n))