import numpy as np
import math


R = 8.314e-3


class CpLandscape(object):
    def __init__(self, T, T1, T2, C1, C2, T_spike, spike_height, spike_width):
        self.T = T
        self.T1 = T1
        self.T2 = T2
        self.C1 = C1
        self.C2 = C2
        self.T_spike = T_spike
        self.spike_height = spike_height
        self.spike_width = spike_width

        self.mean = None
        self.variance = None
        self.x = None
        self._setup_mean_var()
        self.update()

    @property
    def params(self):
        return np.array([self.T, 0.0])

    @params.setter
    def params(self, new_params):
        self.T = new_params[0]
        self._setup_mean_var()

    def update(self):
        self.x = np.random.normal(self.mean, math.sqrt(self.variance))

    def get_energy(self):
        return self.x / (R * self.T)

    def get_derivs(self):
        return np.array([-self.x / (R * self.T**2), 0])

    def get_trial_energy(self, trial_x):
        return trial_x / (R * self.T)

    def get_trial_derivs(self, trial_x):
        return np.array([-trial_x / (R * self.T**2), 0])

    def _setup_mean_var(self):
        self.mean = self._get_enthalpy(self.T)
        self.variance = R * self.T**2 * self._get_heat_capacity(self.T)

    def _get_heat_capacity(self, T):
        delta_C = self.C2 - self.C1
        if T < (self.T_spike - 2 * self.spike_width):
            x = self.C1
        elif T > (self.T_spike + 2 * self.spike_width):
            x = self.C2
        else:
            x = self.C1 + (T - (self.T_spike - 2 * self.spike_width)) / (4 * self.spike_width) * delta_C
        return x + self.spike_height * math.exp(-(T - self.T_spike)**2 / self.spike_width**2)

    def _get_enthalpy(self, T):
        temps = np.linspace(self.T1, T, 500)
        cvs = np.array([self._get_heat_capacity(t) for t in temps])
        enthalpy = self.C1 * self.T1 + np.trapz(cvs, temps)
        return enthalpy
