from __future__ import print_function

import restrainedprotein as rp
import remd
import adapt
import numpy as np
import cPickle as pickle


if __name__ == '__main__':
    n_cond = 8
    temperatures = np.linspace(440.0, 450.0, n_cond)
    temperatures[0] = 300.0
    # temperatures = np.array([300.0, 328.0, 329.0, 330.0])
    force_constants = np.linspace(1.0, 0, n_cond)

    walkers = []
    for i, (t, k) in enumerate(zip(temperatures, force_constants)):
        w = rp.RestrainedProtein(i, 'topol.top', 'inpcrd.crd', t, k, [], [], [])
        print(w.simulation.context.getPlatform().getName())
        walkers.append(w)

    r = remd.RemdLadder(walkers)

    lr = adapt.LearningRateDecay(np.array((4.00, 1e-2)), 0.05)
    m = adapt.MomentumSGD(0.9, adapt.compute_derivative_log_total_acc, lr)
    a = adapt.Adaptor(r, 40, m)

    n_steps = 10000
    for i in range(n_steps):
        a.run(1)

        with open('acceptance.pkl', 'w') as outfile:
            pickle.dump(a.acceptance, outfile)

        with open('params.pkl', 'w') as outfile:
            pickle.dump(a.params, outfile)

        with open('derivs.pkl', 'w') as outfile:
            pickle.dump(m.derivs, outfile)
