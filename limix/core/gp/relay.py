class GPMeanRelay(object):
    def __init__(self, gp):
        self._gp = gp

    # mean can access b from gp
    # through the relay
    def b(self):
        return self._gp.b()
