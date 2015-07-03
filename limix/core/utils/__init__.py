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


def assert_type(arg, type_, param_name):

    err_msg = ("Parameter %s is not of type %s.%s."
               % (param_name, type_.__module__, type_.__name__))

    if type(arg) is not type_:
        raise TypeError(err_msg)


def assert_subtype(arg, type_, param_name):
    err_msg = ("Parameter %s must have %s.%s inheritance."
               % (param_name, type_.__module__, type_.__name__))

    if not issubclass(type(arg), type_):
        raise TypeError(err_msg)


def assert_type_or_list_type(arg, type_, param_name):

    err_msg = ("Parameter %s is not of type "
               "%s.%s nor a list or a tuple of the same."
               % (param_name, type_.__module__, type_.__name__))

    if type(arg) in (list, tuple):
        for a in arg:
            if type(a) is not type_:
                raise TypeError(err_msg)
    else:
        if type(arg) is not type_:
            raise TypeError(err_msg)


def assert_subtype_or_list_subtype(arg, type_, param_name):

    err_msg = ("Parameter %s is not of type "
               "%s.%s nor a list or a tuple of the same."
               % (param_name, type_.__module__, type_.__name__))

    if type(arg) in (list, tuple):
        for a in arg:
            if not issubclass(type(a),  type_):
                raise TypeError(err_msg)
    else:
        if issubclass(type(arg),  type_):
            raise TypeError(err_msg)
