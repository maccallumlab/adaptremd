import numpy as np


def compute_derivative_total_acc(remd):
    acc = remd.acceptance
    A_total = np.product(acc)

    lower_derivs, upper_derivs = remd.derivs

    derivs = np.zeros_like(lower_derivs)
    n_replicas, n_params = derivs.shape

    for i in range(n_replicas):
        if i > 0:
            A = acc[i - 1]
            dA = lower_derivs[i]
            derivs[i] += dA / A

        if i < n_replicas - 1:
            A = acc[i]
            dA = upper_derivs[i]
            derivs[i] += dA / A

    # we set the endpoint derivatives to zeros
    # because they are fixed
    derivs[0, :] = 0
    derivs[-1, :] = 0

    return derivs * A_total


def compute_derivative_log_total_acc(remd):
    acc = remd.acceptance

    lower_derivs, upper_derivs = remd.derivs

    derivs = np.zeros_like(lower_derivs)
    n_replicas, n_params = derivs.shape

    for i in range(n_replicas):
        if i > 0:
            A = acc[i - 1]
            dA = lower_derivs[i]
            derivs[i] += dA / A

        if i < n_replicas - 1:
            A = acc[i]
            dA = upper_derivs[i]
            derivs[i] += dA / A

    # we set the endpoint derivatives to zeros
    # because they are fixed
    derivs[0, :] = 0
    derivs[-1, :] = 0

    return derivs


class LearningRateDecay(object):
    def __init__(self, initial_rate, decay):
        self.initial_rate = np.array(initial_rate)
        self.decay = decay

    def __call__(self, step):
        return self.initial_rate / (1.0 + self.decay * step)


class MomentumSGD(object):
    def __init__(self, momentum, deriv_func, learn_func):
        self.momentum = momentum
        self.learn_func = learn_func
        self.deriv_func = deriv_func
        self.step = 0
        self.v = 0.0
        self.derivs = []

    def update(self, remd):
        params = self._extract_params(remd)
        print params
        derivs = self.deriv_func(remd)
        lr = self.learn_func(self.step)
        self.v = self.momentum * self.v + lr * derivs
        self.derivs.append(self.v)
        params += self.v
        print params
        self._update_params(remd, params)
        self.step += 1

    @staticmethod
    def _extract_params(remd):
        n_reps = remd.n_walkers
        n_params = remd.n_params

        params = np.zeros((n_reps, n_params))
        for i in range(n_reps):
            params[i, :] = remd.walkers[i].params

        return params

    @staticmethod
    def _update_params(remd, params):
        n_reps = params.shape[0]
        for i in range(n_reps):
            remd.walkers[i].params = params[i, :]


class Adaptor(object):
    def __init__(self, remd, n_steps, optimizer):
        self.remd = remd
        self.n_steps = n_steps
        self.optimizer = optimizer
        self.params = []
        self.params.append(self.optimizer._extract_params(self.remd))
        self.acceptance = []

    def run(self, iterations):
        for _ in range(iterations):
            for _ in range(self.n_steps):
                self.remd.update()

            self.optimizer.update(self.remd)
            self.params.append(self.optimizer._extract_params(self.remd))
            self.acceptance.append(self.remd.acceptance)
            self.remd.reset_stats()
