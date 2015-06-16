import inspect
import numpy as np
from limix.core.type.exception import NotArrayConvertibleError

def my_name():
    return inspect.stack()[1][3]

def assert_finite_array(*arrays):
    for a in arrays:
        if not np.isfinite(a).all():
            raise ValueError("Array must not contain infs or NaNs")

def assert_make_float_array(arr, arg_name):
    try:
        arr = np.asarray(arr, dtype=float)
    except ValueError as e:
        raise NotArrayConvertibleError("%s has to be float-array "
                                       "convertible." % arg_name)
    return arr
