import inspect

def my_name():
    return inspect.stack()[1][3]
