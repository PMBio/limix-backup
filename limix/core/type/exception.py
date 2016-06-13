class TooExpensiveOperationError(RuntimeError):
    def __init__(self, msg):
        RuntimeError.__init__(self, msg)

class NotArrayConvertibleError(TypeError):
    def __init__(self, msg):
        TypeError.__init__(self, msg)

class UndefinedInputError(RuntimeError):
    def __init__(self, msg):
        RuntimeError.__init__(self, msg)

if __name__ == '__main__':

    print(('Oi danilo %d'
          'Como voce esta? %d' % (1, 2)))

    def expensive():
        raise TooExpensiveOperationError('ola')

    try:
        expensive()
    except TooExpensiveOperationError as e:
        print(e)
