from __future__ import print_function

import restrainedprotein as rp
import remd
import adapt
import numpy as np
import cPickle as pickle

n_cond = 4
# temperatures = np.linspace(300.0, 350.0, n_cond)
temperatures = np.array([300.0, 328.0, 329.0, 330.0])
force_constants = np.linspace(1.0, 0, n_cond)

walkers = []
for i, (t, k) in enumerate(zip(temperatures, force_constants)):
    w = rp.RestrainedProtein(i, 'topol.top', 'inpcrd.crd', t, k, [], [], [])
    walkers.append(w)

r = remd.RemdLadder(walkers)

lr = adapt.LearningRateDecay(np.array((1.0, 1e-2)), 0.001)
m = adapt.MomentumSGD(0.9, adapt.compute_derivative_log_total_acc, lr)
a = adapt.Adaptor(r, 10, m)

n_steps = 10000
for i in range(n_steps):
    a.run(1)

    with open('acceptance.pkl', 'w') as outfile:
        pickle.dump(r.acceptance, outfile)

    with open('params.pkl', 'w') as outfile:
        pickle.dump(a.params, outfile)

    with open('derivs.pkl', 'w') as outfile:
        pickle.dump(m.derivs, outfile)
