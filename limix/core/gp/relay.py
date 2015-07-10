class GPMeanRelay(object):
    def __init__(self, gp):
        self._gp = gp

    # Mean talk to GP to help update
    # itself.
    def _update_mean(self):
        self._gp.update_b()
