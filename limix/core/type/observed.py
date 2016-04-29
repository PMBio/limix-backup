# USAGE: Please, look at the end of the file for an example.

class Observed(object):
    def __init__(self):
        self._init_interesteds()

    def _init_interesteds(self):
        self._interesteds = dict()

    def register(self, interested, event='default'):
        if not hasattr(self, '_interesteds'):
            self._init_interesteds()
        if event not in list(self._interesteds.keys()):
            self._interesteds[event] = []
        self._interesteds[event].append(interested)

    def _notify(self, event='default'):
        if not hasattr(self, '_interesteds'):
            self._init_interesteds()
        if event in self._interesteds:
            for i in self._interesteds[event]:
                i()

if __name__ == '__main__':
    class A(Observed):
        def set_params(self):
            self._notify()

        def trigger_event1(self):
            self._notify('event1')

    class B(object):
        def __init__(self, a):
            a.register(self.call_me)
            a.register(self.call_me_on_event1, 'event1')

        def call_me(self):
            print('call_me')

        def call_me_on_event1(self):
            print('call_me_on_event1')

    a = A()
    b = B(a)
    a.set_params() # prints call_me
    a.set_params() # prints call_me
    a.trigger_event1() # prints call_me_on_event1
