def msg_too_expensive_dim(method_name, dim):
    return('%s method is too expensive to be '
           'run for matrices with dimension '
           'greater than %d.' % (method_name, dim))
