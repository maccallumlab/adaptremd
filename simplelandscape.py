import numpy as np
import numba


@numba.jit
def flat_energy(xs):
    return np.zeros_like(xs)

@numba.jit
def linear_energy(xs):
    return 0.01 * xs


@numba.jit
def logistic_energy(xs):
    return 50.0 / (1 + np.exp(-0.04 * (xs - 500)))


@numba.jit
def quadratic_energy(xs):
    return 0.5 * 1e-3 * (xs - 500)**2


@numba.jit
def umbrella_bias(xs, params):
    out = np.zeros((xs.shape[0], 3))
    x0, k = params
    out[:, 0] = 0.5 * k * (xs - x0)**2
    out[:, 1] = k * (x0 - xs)
    out[:, 2] = 0.5 * (xs - x0)**2
    return out


class OneDimLandscape(object):
    def __init__(self, energy_func, bias_func, x0, param0, nx=1000):
        self._energy_func = energy_func
        self._bias_func = bias_func
        self.x = x0
        self._params = param0.copy()
        self._n_params = param0.shape[0]
        self._nx = nx
        self._xs = np.array(range(nx))
        self._landscape = None
        self._derivs = None
        self._probs = None

        self._update_landscape()

    @property
    def nx(self):
        return self._nx

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, new_params):
        self._params = new_params
        self._update_landscape()

    def update(self):
        self.x = np.random.choice(self._xs, p=self._probs)

    def get_energy(self):
        return self._landscape[self.x]

    def get_derivs(self):
        return self._derivs[self.x, :]

    def get_trial_energy(self, trial_x):
        return self._landscape[trial_x]

    def get_trial_derivs(self, trial_x):
        return self._derivs[trial_x, :]

    def _update_landscape(self):
        #energy = np.array([self._energy_func(x) for x in self._xs])
        energy = self._energy_func(self._xs)
        bias = self._bias_func(self._xs, self._params)
        # bias = np.array([self._bias_func(x, self._params, self._nx) for x in self._xs])
        bias_e = bias[:, 0]
        bias_deriv = bias[:, 1:]
        self._landscape = energy + bias_e
        self._derivs = bias_deriv
        self._update_probs()

    def _update_probs(self):
        self._probs = np.exp(-self._landscape)
        self._probs /= np.sum(self._probs)
