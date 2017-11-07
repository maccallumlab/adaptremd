import numpy as np
import math


R = 8.314e-3


class HarmonicLandscape(object):
    def __init__(self, T, k=1.0):
        self.T = T
        self.k = k
        self.std = None

        self._compute_std()
        self.update()


    @property
    def params(self):
        return np.array([self.T])

    @params.setter
    def params(self, new_params):
        self.T = new_params[0]
        self._compute_std()

    def update(self):
        self.x = np.random.normal(0, self.std)

    def get_energy(self):
        return self.k * self.x**2 / R / self.T

    def get_derivs(self):
        return -self.k * self.x**2 / R / self.T**2

    def get_trial_energy(self, trial_x):
        return self.k * trial_x**2 / R / self.T

    def get_trial_derivs(self, trial_x):
        return -self.k * trial_x**2 / R / self.T**2

    def _compute_std(self):
        self.std = math.sqrt(R * self.T / 2.0 / self.k)
