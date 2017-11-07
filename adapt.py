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
            dA = lower_derivs[i, :]
            derivs[i, :] += dA / A * A_total

        if i < n_replicas - 1:
            A = acc[i]
            dA = upper_derivs[i, :]
            derivs[i, :] += dA / A * A_total

    # we set the endpoint derivatives to zeros
    # because they are fixed
    derivs[0, :] = 0
    derivs[-1, :] = 0

    return derivs


def compute_derivative_log_total_acc(remd):
    acc = remd.acceptance + remd.eps

    lower_derivs, upper_derivs = remd.derivs

    derivs = np.zeros_like(lower_derivs)
    n_replicas, n_params = derivs.shape

    for i in range(n_replicas):
        if i > 0:
            A = acc[i - 1]
            dA = lower_derivs[i, :]
            derivs[i, :] += dA / A

        if i < n_replicas - 1:
            A = acc[i]
            dA = upper_derivs[i, :]
            derivs[i, :] += dA / A

    # we set the endpoint derivatives to zeros
    # because they are fixed
    derivs[0, :] = 0
    derivs[-1, :] = 0

    return derivs


def compute_derivative_log_total_acc_pen(remd):
    alpha1 = 1.0
    alpha2 = 1.0

    acc = remd.acceptance + 1e-6

    lower_derivs, upper_derivs = remd.derivs

    derivs = np.zeros_like(lower_derivs)
    n_replicas, n_params = derivs.shape

    eps = 0
    for i in range(n_replicas):
        if i > 0:
            A = acc[i - 1] + eps
            dA = lower_derivs[i]
            x = alpha1
            y = 2. * alpha2 * (np.log(A) - np.log(np.mean(acc)))
            derivs[i] += (x - y) * dA / A

        if i < n_replicas - 1:
            A = acc[i] + eps
            dA = upper_derivs[i]
            x = alpha1
            y = 2. * alpha2 * (np.log(A) - np.log(np.mean(acc)))
            derivs[i] += (x - y) * dA / A

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
    def __init__(self, momentum, deriv_func, learn_func, param_bounds):
        self.momentum = momentum
        self.learn_func = learn_func
        self.deriv_func = deriv_func
        self.param_bounds = param_bounds
        self.step = 0
        self.v = 0.0
        self.derivs = []
        self.vs = []

    def update(self, remd):
        params = self._extract_params(remd)
        derivs = self.deriv_func(remd)
        lr = self.learn_func(self.step)
        self.v = self.momentum * self.v + lr * derivs
        self.derivs.append(derivs)
        self.vs.append(self.v)
        params += self.v

        # make sure the parameters don't go outside of the specified bounds
        params = np.maximum(params, self.param_bounds[0, :])
        params = np.minimum(params, self.param_bounds[1, :])

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


class MomentumSGD2(object):
    def __init__(self, momentum, deriv_func, learn_func, param_bounds):
        self.momentum = momentum
        self.learn_func = learn_func
        self.deriv_func = deriv_func
        self.param_bounds = param_bounds
        self.step = 0
        self.v = 0.0
        self.derivs = []

    def update(self, remd):
        params = self._extract_params(remd)
        derivs = self.deriv_func(remd)
        lr = self.learn_func(self.step)
        self.v = self.momentum * self.v + lr * derivs
        self.derivs.append(self.v)
        params += self.v

        # make sure the parameters don't go outside of the specified bounds
        params = np.maximum(params, self.param_bounds[0, :])
        params = np.minimum(params, self.param_bounds[1, :])

        self._update_params(remd, params)
        self.step += 1

    @staticmethod
    def _extract_params(remd):
        return remd.params

    @staticmethod
    def _update_params(remd, params):
        remd.params = params


class Adam(object):
    def __init__(self, gamma1, gamma2, deriv_func, learn_func, param_bounds):
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.learn_func = learn_func
        self.deriv_func = deriv_func
        self.param_bounds = param_bounds
        self.step = 0
        self.m = 0.0
        self.g = 0.0
        self.derivs = []
        self.vs = []

    def update(self, remd):
        params = self._extract_params(remd)
        derivs = self.deriv_func(remd)
        lr = self.learn_func(self.step)

        self.m = self.gamma1 * self.m + (1.0 - self.gamma1) * derivs
        self.g = self.gamma2 * self.g + (1.0 - self.gamma2) * derivs**2

        mhat = self.m / (1.0 - self.gamma1**(self.step + 1))
        ghat = self.g / (1.0 - self.gamma2**(self.step + 1))

        v = lr * mhat / (np.sqrt(ghat) + 1e-8)
        self.vs.append(v)
        self.derivs.append(derivs)
        params += v

        params = np.maximum(params, self.param_bounds[0, :])
        params = np.minimum(params, self.param_bounds[1, :])

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


class Adam2(object):
    def __init__(self, gamma1, gamma2, deriv_func, learn_func, param_bounds):
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.learn_func = learn_func
        self.deriv_func = deriv_func
        self.param_bounds = param_bounds
        self.step = 0
        self.m = 0.0
        self.g = 0.0
        self.derivs = []
        self.vs = []

    def update(self, remd):
        params = self._extract_params(remd)
        derivs = self.deriv_func(remd)
        lr = self.learn_func(self.step)

        self.m = self.gamma1 * self.m + (1.0 - self.gamma1) * derivs
        self.g = self.gamma2 * self.g + (1.0 - self.gamma2) * derivs**2

        mhat = self.m / (1.0 - self.gamma1**(self.step + 1))
        ghat = self.g / (1.0 - self.gamma2**(self.step + 1))

        v = lr * mhat / (np.sqrt(ghat) + 1e-8)
        self.vs.append(v)
        self.derivs.append(derivs)

        params += v
        # make sure the parameters don't go outside of the specified bounds
        params = np.maximum(params, self.param_bounds[0, :])
        params = np.minimum(params, self.param_bounds[1, :])

        self._update_params(remd, params)
        self.step += 1

    @staticmethod
    def _extract_params(remd):
        return remd.params

    @staticmethod
    def _update_params(remd, params):
        remd.params = params


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
            self.remd.reset_stats()
            for _ in range(self.n_steps):
                self.remd.update()

            self.optimizer.update(self.remd)
            self.params.append(self.optimizer._extract_params(self.remd))
            self.acceptance.append(self.remd.acceptance)
