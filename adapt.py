import numpy as np
<<<<<<< 71ef6c480f52062720d83ce6ed479d90fb53bd7b
from collections import namedtuple
=======
import itertools
>>>>>>> Updated plots


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
    def __init__(self, gamma1, gamma2, deriv_func, param_bounds):
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.deriv_func = deriv_func
        self.param_bounds = param_bounds
        self.step = 0
        self.m = 0.0
        self.g = 0.0
        self.derivs = []
        self.vs = []

    def update(self, remd, lr):
        params = self._extract_params(remd)
        derivs = self.deriv_func(remd)

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


AdaptationStep = namedtuple('AdaptationStep', 'step burn_in_only output_only update learning_rate')


class AdaptationIter(object):
    def __init__(self, max_steps, discard_first_steps,
                 init_cycle_length, cycle_length_doubling_cycles, fraction_batch_discard,
                 learning_rate_decay_cycles, init_learning_rate):
        self.current_step = 1
        self.current_cycle = 0
        self.state = None
        self.result = None
        self.target = None

        self.max_steps = max_steps
        self.discard_first_steps = discard_first_steps
        self.initial_cycle_length = init_cycle_length
        self.cycle_length_base = 2.0 ** (1.0 / cycle_length_doubling_cycles)
        self.fraction_batch_discard = fraction_batch_discard
        self.learning_rate_decay_cycles = learning_rate_decay_cycles
        self.init_learning_rate = init_learning_rate

        self.transition_to_init()

    def __iter__(self):
        return self

    @property
    def current_cycle_length(self):
        return int(round(self.initial_cycle_length * self.cycle_length_base ** self.current_cycle))

    @property
    def current_learning_rate(self):
        return self.init_learning_rate / (1.0 + (1.0 / self.learning_rate_decay_cycles) * self.current_cycle)

    def next(self):
        self.handle_step()
        self.current_step += 1
        return self.result

    def handle_step(self):
        if self.state == 'init':
            self.handle_step_init()
        if self.state == 'init_final':
            self.handle_step_init_final()
        elif self.state == 'burning':
            self.handle_step_burning()
        elif self.state == 'burn':
            self.handle_step_burn()
        elif self.state == 'updating':
            self.handle_step_updating()
        elif self.state == 'update':
            self.handle_step_update()
        elif self.state == 'finalizing':
            self.handle_step_finalizing()
        elif self.state == 'final':
            self.handle_step_final()
        elif self.state == 'done':
            self.handle_step_done()

    def transition_to_init(self):
        if self.discard_first_steps == 0:
            self.transition_to_burning(offset=1)
        else:
            self.state = 'init'
            self.target = self.discard_first_steps

    def handle_step_init(self):
        self.result = AdaptationStep(self.current_step, False, False, False, None)
        if self.current_step == self.target:
            self.transition_to_init_final()

    def transition_to_init_final(self):
        self.state = 'init_final'

    def handle_step_init_final(self):
        self.result = AdaptationStep(self.current_step, True, False, False, None)
        self.transition_to_burning()

    def transition_to_burning(self, offset=0):
        self.state = 'burning'
        n_burn = int(self.fraction_batch_discard * self.current_cycle_length)
        if n_burn == 0:
            self.transition_to_updating(offset)
        else:
            self.target = self.current_step + n_burn - offset - 1

    def handle_step_burning(self):
        self.result = AdaptationStep(self.current_step, False, False, False, None)
        if self.current_step == self.target:
            self.transition_to_burn()

    def transition_to_burn(self):
        self.state = 'burn'

    def handle_step_burn(self):
        self.result = AdaptationStep(self.current_step, True, False, False, None)
        self.transition_to_updating()

    def transition_to_updating(self, offset=0):
        self.state = 'updating'
        n_burn = int(self.fraction_batch_discard * self.current_cycle_length)
        n_update = self.current_cycle_length - n_burn
        self.target = self.current_step + n_update - offset - 1

    def handle_step_updating(self):
        self.result = AdaptationStep(self.current_step, False, False, False, None)
        if self.current_step == self.target:
            self.transition_to_update()

    def transition_to_update(self):
        self.state = 'update'

    def handle_step_update(self):
        self.result = AdaptationStep(self.current_step, False, False, True, self.current_learning_rate)
        self.current_cycle += 1
        if self.current_step + self.current_cycle_length >= self.max_steps - 2:
            self.transition_to_finalizing()
        else:
            self.transition_to_burning()

    def transition_to_finalizing(self):
        self.state = 'finalizing'

    def handle_step_finalizing(self):
        self.result = AdaptationStep(self.current_step, False, False, False, None)
        if self.current_step == self.max_steps - 1:
            self.transition_to_final()

    def transition_to_final(self):
        self.state = 'final'

    def handle_step_final(self):
        self.result = AdaptationStep(self.current_step, False, True, False, None)
        self.transition_to_done()

    def transition_to_done(self):
        self.state = 'done'

    def handle_step_done(self):
        raise StopIteration

class ExponentialAdaptor(object):
    def __init__(self, remd, eta_init, gamma, optimizer):
        self.remd = remd
        self.eta_init = float(eta_init)
        self.gamma = float(gamma)
        self.current_step = 0
        self.optimizer = optimizer
        self.params = []
        self.params.append(self.optimizer._extract_params(self.remd))
        self.acceptance = []
        self.adapted_at = [0]

    def run(self, iterations):
        for iteration in range(iterations):
            self.remd.reset_stats()
            for _ in range(self._compute_steps(iteration)):
                self.remd.update()
            self.current_step += self._compute_steps(iteration)

            self.optimizer.update(self.remd)
            self.params.append(self.optimizer._extract_params(self.remd))
            self.acceptance.append(self.remd.acceptance)
            self.adapted_at.append(self.current_step)

    def min_iterations_for_target_steps(self, target):
        steps = 0
        for i in itertools.count():
            steps += self._compute_steps(i)
            if steps > target:
                break
        return i

    def _compute_steps(self, iteration):
        return int(self.eta_init * self.gamma ** iteration)
