import numpy as np
import math
import random


class RemdLadder(object):
    def __init__(self, walkers, n_exchanges=None):
        self.walkers = walkers
        self.n_walkers = len(walkers)
        self.n_params = walkers[0].params.shape[0]
        self.n_exchanges = (self.n_walkers**2 if n_exchanges is None
                            else n_exchanges)

        # keep track of running averages
        self.n_trials = None
        self._acc = None
        self._dAupper = None
        self._dAlower = None
        self.reset_stats()

    def reset_stats(self):
        self.n_trials = 0
        self._acc = np.zeros(self.n_walkers - 1)
        self._dA_upper = np.zeros((self.n_walkers, self.n_params))
        self._dA_lower = np.zeros((self.n_walkers, self.n_params))
        self._dE_A_lower = np.zeros((self.n_walkers, self.n_params))
        self._dE_A_upper = np.zeros((self.n_walkers, self.n_params))
        self._dE_upper = np.zeros((self.n_walkers, self.n_params))
        self._dE_lower = np.zeros((self.n_walkers, self.n_params))
        self._A_upper = np.zeros(self.n_walkers)
        self._A_lower = np.zeros(self.n_walkers)

    def update(self):
        self.n_trials += 1

        for w in self.walkers:
            w.update()


        for _ in range(self.n_exchanges):
            i = random.choice(range(self.n_walkers - 1))
            j = i + 1

            accept_prob = self._compute_acceptance(i, j)
            accept = (True if random.random() <= accept_prob
                      else False)

            if accept:
                w1 = self.walkers[i]
                w2 = self.walkers[j]
                w1.x, w2.x = w2.x, w1.x

        self._update_stats()

    @property
    def acceptance(self):
        return self._acc / self.n_trials

    @property
    def derivs(self):
        upper = (self._dA_upper / self.n_trials -
                 self._dE_A_upper / self.n_trials +
                 self._dE_upper / self.n_trials *
                 self._A_upper[:, np.newaxis] / self.n_trials)
        lower = (self._dA_lower / self.n_trials -
                 self._dE_A_lower / self.n_trials +
                 self._dE_lower / self.n_trials *
                 self._A_lower[:, np.newaxis] / self.n_trials)
        return (lower, upper)

    def _compute_acceptance(self, i, j):
        w1 = self.walkers[i]
        w2 = self.walkers[j]

        e11 = w1.get_energy()
        e22 = w2.get_energy()
        e12 = w1.get_trial_energy(w2.x)
        e21 = w2.get_trial_energy(w1.x)

        delta = e12 + e21 - e11 - e22
        if delta < 0:
            accept = 1.0
        else:
            accept = math.exp(-delta)

        return accept

    def _update_stats(self):
        self._update_acc()
        self._update_derivs()

    def _update_acc(self):
        for i in range(self.n_walkers - 1):
            self._acc[i] += self._compute_acceptance(i, i+1)

    def _update_derivs(self):
        for i in range(self.n_walkers):
            for upper in [False, True]:
                j = i + 1 if upper else i - 1
                if j < 0 or j >= self.n_walkers:
                    continue

                wi = self.walkers[i]
                wj = self.walkers[j]

                acc = self._compute_acceptance(i, j)
                dE = wi.get_derivs()

                if acc == 1.0:
                    dA = np.zeros_like(dE)
                else:
                    dA = acc * (wi.get_trial_derivs(wi.x) -
                                wi.get_trial_derivs(wj.x))

                if upper:
                    self._dA_upper[i, :] += dA
                    self._dE_A_upper[i, :] += acc * dE
                    self._dE_upper[i, :] += dE
                    self._A_upper[i] += acc
                else:
                    self._dA_lower[i, :] += dA
                    self._dE_A_lower[i, :] += acc * dE
                    self._dE_lower[i, :] += dE
                    self._A_lower[i] += acc



class RemdLadderJensen(object):
    def __init__(self, walkers, n_exchanges=None):
        self.walkers = walkers
        self.n_walkers = len(walkers)
        self.n_params = walkers[0].params.shape[0]
        self.n_exchanges = (self.n_walkers**2 if n_exchanges is None
                            else n_exchanges)

        # keep track of running averages
        self.n_trials = None
        self._acc = None
        self.reset_stats()

    def reset_stats(self):
        self.n_trials = 0
        self._acc = np.zeros(self.n_walkers - 1)
        self._d_lnA_upper = np.zeros((self.n_walkers, self.n_params))
        self._d_lnA_lower = np.zeros((self.n_walkers, self.n_params))
        self._dE_lnA_lower = np.zeros((self.n_walkers, self.n_params))
        self._dE_lnA_upper = np.zeros((self.n_walkers, self.n_params))
        self._dE_upper = np.zeros((self.n_walkers, self.n_params))
        self._dE_lower = np.zeros((self.n_walkers, self.n_params))
        self._lnA_upper = np.zeros(self.n_walkers)
        self._lnA_lower = np.zeros(self.n_walkers)

    def update(self):
        self.n_trials += 1

        for w in self.walkers:
            w.update()


        for _ in range(self.n_exchanges):
            i = random.choice(range(self.n_walkers - 1))
            j = i + 1

            accept_prob = self._compute_acceptance(i, j)
            accept = (True if random.random() <= accept_prob
                      else False)

            if accept:
                w1 = self.walkers[i]
                w2 = self.walkers[j]
                w1.x, w2.x = w2.x, w1.x

        self._update_stats()

    @property
    def acceptance(self):
        return self._acc / self.n_trials

    @property
    def ln_A(self):
        return (self._lnA_lower / self.n_trials,
                self._lnA_upper / self.n_trials)
    @property
    def derivs(self):
        upper = (self._d_lnA_upper / self.n_trials -
                 self._dE_lnA_upper / self.n_trials +
                 self._dE_upper / self.n_trials *
                 self._lnA_upper[:, np.newaxis] / self.n_trials)
        lower = (self._d_lnA_lower / self.n_trials -
                 self._dE_lnA_lower / self.n_trials +
                 self._dE_lower / self.n_trials *
                 self._lnA_lower[:, np.newaxis] / self.n_trials)
        N = 1
        upper *= N * (-1)**(N+1) * (self._lnA_upper[:, np.newaxis] / self.n_trials)**(N-1)
        lower *= N * (-1)**(N+1) * (self._lnA_lower[:, np.newaxis] / self.n_trials)**(N-1)
        return (lower, upper)

    def _compute_acceptance(self, i, j):
        w1 = self.walkers[i]
        w2 = self.walkers[j]

        e11 = w1.get_energy()
        e22 = w2.get_energy()
        e12 = w1.get_trial_energy(w2.x)
        e21 = w2.get_trial_energy(w1.x)

        delta = e12 + e21 - e11 - e22
        if delta < 0:
            accept = 1.0
        else:
            accept = math.exp(-delta)

        return accept

    def _compute_ln_acceptance(self, i, j):
        w1 = self.walkers[i]
        w2 = self.walkers[j]

        e11 = w1.get_energy()
        e22 = w2.get_energy()
        e12 = w1.get_trial_energy(w2.x)
        e21 = w2.get_trial_energy(w1.x)
        delta = e12 + e21 - e11 - e22

        return min(0, -delta)

    def _update_stats(self):
        self._update_acc()
        self._update_derivs()

    def _update_acc(self):
        for i in range(self.n_walkers - 1):
            self._acc[i] += self._compute_acceptance(i, i+1)

    def _update_derivs(self):
        for i in range(self.n_walkers):
            for upper in [False, True]:
                j = i + 1 if upper else i - 1
                if j < 0 or j >= self.n_walkers:
                    continue

                wi = self.walkers[i]
                wj = self.walkers[j]

                lnA = self._compute_ln_acceptance(i, j)
                dE = wi.get_derivs()

                if lnA == 0.0:
                    d_lnA = np.zeros_like(dE)
                else:
                    d_lnA = (wi.get_trial_derivs(wi.x) -
                             wi.get_trial_derivs(wj.x))

                if upper:
                    self._d_lnA_upper[i, :] += d_lnA
                    self._dE_lnA_upper[i, :] += lnA * dE
                    self._dE_upper[i, :] += dE
                    self._lnA_upper[i] += lnA
                else:
                    self._d_lnA_lower[i, :] += d_lnA
                    self._dE_lnA_lower[i, :] += lnA * dE
                    self._dE_lower[i, :] += dE
                    self._lnA_lower[i] += lnA



class RemdLadder2(object):
    def __init__(self, init_params, n_exchanges=None):
        self.params = init_params
        self.n_walkers = init_params.shape[0]
        self.n_params = init_params.shape[1]

        self.n_exchanges = (self.n_walkers**2 if n_exchanges is None
                            else n_exchanges)

        # keep track of running averages
        self.n_trials = None
        self._acc = None
        self._dAupper = None
        self._dAlower = None
        self.reset_stats()

    def reset_stats(self):
        self.n_trials = 0
        self._acc = np.zeros(self.n_walkers - 1)
        self._dA_upper = np.zeros((self.n_walkers, self.n_params))
        self._dA_lower = np.zeros((self.n_walkers, self.n_params))
        self._dE_A_lower = np.zeros((self.n_walkers, self.n_params))
        self._dE_A_upper = np.zeros((self.n_walkers, self.n_params))
        self._dE_upper = np.zeros((self.n_walkers, self.n_params))
        self._dE_lower = np.zeros((self.n_walkers, self.n_params))
        self._A_upper = np.zeros(self.n_walkers)
        self._A_lower = np.zeros(self.n_walkers)

    def update(self, energy_matrix, deriv_matrix):
        self.n_trials += 1
        perm = list(range(energy_matrix.shape[0]))

        for _ in range(self.n_exchanges):
            i = random.choice(range(self.n_walkers - 1))
            j = i + 1

            accept_prob = self._compute_acceptance(i, j, energy_matrix)
            accept = (True if random.random() <= accept_prob
                      else False)

            if accept:
                perm[i], perm[j] = perm[j], perm[i]
                energy_matrix[i, :], energy_matrix[j, :] = energy_matrix[j, :], energy_matrix[i, :]
                energy_matrix[:, i], energy_matrix[:, j] = energy_matrix[:, j], energy_matrix[:, i]
                deriv_matrix[i, :], deriv_matrix[j, :] = deriv_matrix[j, :], deriv_matrix[i, :]
                deriv_matrix[:, i], deriv_matrix[:, j] = deriv_matrix[:, j], deriv_matrix[:, i]

        self._update_stats(energy_matrix, deriv_matrix)
        return perm

    @property
    def acceptance(self):
        return self._acc / self.n_trials

    @property
    def derivs(self):
        upper = (self._dA_upper / self.n_trials -
                 self._dE_A_upper / self.n_trials +
                 self._dE_upper / self.n_trials *
                 self._A_upper[:, np.newaxis] / self.n_trials)
        lower = (self._dA_lower / self.n_trials -
                 self._dE_A_lower / self.n_trials +
                 self._dE_lower / self.n_trials *
                 self._A_lower[:, np.newaxis] / self.n_trials)
        return (lower, upper)

    def _compute_acceptance(self, i, j, energy_matrix):
        e11 = energy_matrix[i, i]
        e22 = energy_matrix[j, j]
        e12 = energy_matrix[i, j]
        e21 = energy_matrix[j, i]

        delta = e12 + e21 - e11 - e22
        if delta < 0:
            accept = 1.0
        else:
            accept = math.exp(-delta)

        return accept

    def _update_stats(self, energy_matrix, deriv_matrix):
        self._update_acc(energy_matrix)
        self._update_derivs(energy_matrix, deriv_matrix)

    def _update_acc(self, energy_matrix):
        for i in range(self.n_walkers - 1):
            self._acc[i] += self._compute_acceptance(i, i+1, energy_matrix)

    def _update_derivs(self, energy_matrix, deriv_matrix):
        for i in range(self.n_walkers):
            for upper in [False, True]:
                j = i + 1 if upper else i - 1
                if j < 0 or j >= self.n_walkers:
                    continue

                acc = self._compute_acceptance(i, j, energy_matrix)
                dE = deriv_matrix[i, i]

                if acc == 1.0:
                    dA = np.zeros_like(dE)
                else:
                    dA = acc * (deriv_matrix[i, i] - deriv_matrix[i, j])

                if upper:
                    self._dA_upper[i, :] += dA
                    self._dE_A_upper[i, :] += acc * dE
                    self._dE_upper[i, :] += dE
                    self._A_upper[i] += acc
                else:
                    self._dA_lower[i, :] += dA
                    self._dE_A_lower[i, :] += acc * dE
                    self._dE_lower[i, :] += dE
                    self._A_lower[i] += acc


class RemdLadderJensen2(object):
    def __init__(self, init_params, n_exchanges=None):
        self.params = init_params
        self.n_walkers = init_params.shape[0]
        self.n_params = init_params.shape[1]

        self.n_exchanges = (self.n_walkers**2 if n_exchanges is None
                            else n_exchanges)

        # keep track of running averages
        self.n_trials = None
        self._acc = None
        self.reset_stats()

    def reset_stats(self):
        self.n_trials = 0
        self._acc = np.zeros(self.n_walkers - 1)
        self._d_lnA_upper = np.zeros((self.n_walkers, self.n_params))
        self._d_lnA_lower = np.zeros((self.n_walkers, self.n_params))
        self._dE_lnA_lower = np.zeros((self.n_walkers, self.n_params))
        self._dE_lnA_upper = np.zeros((self.n_walkers, self.n_params))
        self._dE_upper = np.zeros((self.n_walkers, self.n_params))
        self._dE_lower = np.zeros((self.n_walkers, self.n_params))
        self._lnA_upper = np.zeros(self.n_walkers)
        self._lnA_lower = np.zeros(self.n_walkers)

    def update(self, energy_matrix, deriv_matrix):
        self.n_trials += 1
        perm = list(range(energy_matrix.shape[0]))

        for _ in range(self.n_exchanges):
            i = random.choice(range(self.n_walkers - 1))
            j = i + 1

            accept_prob = self._compute_acceptance(i, j, energy_matrix)
            accept = (True if random.random() <= accept_prob
                      else False)

            if accept:
                perm[i], perm[j] = perm[j], perm[i]
                energy_matrix[i, :], energy_matrix[j, :] = energy_matrix[j, :], energy_matrix[i, :]
                energy_matrix[:, i], energy_matrix[:, j] = energy_matrix[:, j], energy_matrix[:, i]
                deriv_matrix[i, :], deriv_matrix[j, :] = deriv_matrix[j, :], deriv_matrix[i, :]
                deriv_matrix[:, i], deriv_matrix[:, j] = deriv_matrix[:, j], deriv_matrix[:, i]

        self._update_stats(energy_matrix, deriv_matrix)
        return perm

    @property
    def acceptance(self):
        return self._acc / self.n_trials

    @property
    def derivs(self):
        upper = (self._d_lnA_upper / self.n_trials -
                 self._dE_lnA_upper / self.n_trials +
                 self._dE_upper / self.n_trials *
                 self._lnA_upper[:, np.newaxis] / self.n_trials)
        lower = (self._d_lnA_lower / self.n_trials -
                 self._dE_lnA_lower / self.n_trials +
                 self._dE_lower / self.n_trials *
                 self._lnA_lower[:, np.newaxis] / self.n_trials)
        N = 1
        upper *= N * (-1)**(N+1) * (self._lnA_upper[:, np.newaxis] / self.n_trials)**(N-1)
        lower *= N * (-1)**(N+1) * (self._lnA_lower[:, np.newaxis] / self.n_trials)**(N-1)
        return (lower, upper)

    def _compute_acceptance(self, i, j, energy_matrix):
        e11 = energy_matrix[i, i]
        e22 = energy_matrix[j, j]
        e12 = energy_matrix[i, j]
        e21 = energy_matrix[j, i]

        delta = e12 + e21 - e11 - e22
        if delta < 0:
            accept = 1.0
        else:
            accept = math.exp(-delta)

        return accept

    def _compute_ln_acceptance(self, i, j, energy_matrix):
        e11 = energy_matrix[i, i]
        e22 = energy_matrix[j, j]
        e12 = energy_matrix[i, j]
        e21 = energy_matrix[j, i]
        delta = e12 + e21 - e11 - e22

        return min(0, -delta)

    def _update_stats(self, energy_matrix, deriv_matrix):
        self._update_acc(energy_matrix)
        self._update_derivs(energy_matrix, deriv_matrix)

    def _update_acc(self, energy_matrix):
        for i in range(self.n_walkers - 1):
            self._acc[i] += self._compute_acceptance(i, i+1, energy_matrix)

    def _update_derivs(self, energy_matrix, deriv_matrix):
        for i in range(self.n_walkers):
            for upper in [False, True]:
                j = i + 1 if upper else i - 1
                if j < 0 or j >= self.n_walkers:
                    continue

                lnA = self._compute_ln_acceptance(i, j, energy_matrix)
                dE = deriv_matrix[i, i]

                if lnA == 0.0:
                    d_lnA = np.zeros_like(dE)
                else:
                    d_lnA = (deriv_matrix[i, i] -
                             deriv_matrix[i, j])

                if upper:
                    self._d_lnA_upper[i, :] += d_lnA
                    self._dE_lnA_upper[i, :] += lnA * dE
                    self._dE_upper[i, :] += dE
                    self._lnA_upper[i] += lnA
                else:
                    self._d_lnA_lower[i, :] += d_lnA
                    self._dE_lnA_lower[i, :] += lnA * dE
                    self._dE_lower[i, :] += dE
                    self._lnA_lower[i] += lnA


