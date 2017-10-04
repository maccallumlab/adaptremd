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
