from sklearn.base import BaseEstimator

import numpy as np
import scipy.sparse as sp
import statistics

from ctypes import *
from os import path
from sys import platform

dirname = path.dirname(path.abspath(__file__))

if platform == "linux" or platform == "linux2":
    shared_library_name = "libFedTree.so"
elif platform == "win32":
    shared_library_name = "libFedTree.dll"
elif platform == "darwin":
    shared_library_name = "libFedTree.dylib"
else:
    raise EnvironmentError("OS not supported!")

if path.exists(path.abspath(path.join(dirname, shared_library_name))):
    lib_path = path.abspath(path.join(dirname, shared_library_name))
else:
    lib_path = path.join(dirname, "../../build/lib", shared_library_name)

if path.exists(lib_path):
    FedTree = CDLL(lib_path)
else:
    raise RuntimeError("Please build the library first!")

