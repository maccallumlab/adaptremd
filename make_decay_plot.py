#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as pp
from harmonic import HarmonicLandscape
from remd import RemdLadder
import adapt


n_cond = 4
init_temperatures = [10., 5000., 5000., 10000.]

T2 = 10 * (10000 / 10) ** (1.0 / 3.0)
T3 = 10 * (10000 / 10) ** (2.0 / 3.0)
decays = list(reversed([1e-2, 0.5e-2, 0.25e-2, 0.125e-2]))

fig, axes = pp.subplots(1, len(decays), sharex=True, sharey=True, figsize=(6, 2))

for i, decay in enumerate(decays):
    walkers = []
    for t in init_temperatures:
        w = HarmonicLandscape(t)
        walkers.append(w)

    r = RemdLadder(walkers)

    lr = adapt.LearningRateDecay(np.array((50.0,)), decay)
    param_bounds = np.array([[10], [10000]])
    m = adapt.Adam(0.9,
                   0.9,
                   adapt.compute_derivative_log_total_acc,
                   lr,
                   param_bounds)
    a = adapt.ExponentialAdaptor(r, 64, 1.0, m)

    target_iter = 100000
    iterations = a.min_iterations_for_target_steps(target_iter)
    print iterations
    a.run(iterations)

    params = np.array(a.params)
    axes[i].plot(a.adapted_at, params[:, 1:3, 0], linewidth=2)
    axes[i].axhline(T2, color='white', linewidth=2)
    axes[i].axhline(T2, color='black', linewidth=1, linestyle='--')
    axes[i].axhline(T3, color='white', linewidth=2)
    axes[i].axhline(T3, color='black', linewidth=1, linestyle='--')
    axes[i].set_ylim(0, 5000)
    axes[i].set_xlim(0, target_iter)

# set the labels
axes[0].set_ylabel('$x_0$')
for i in range(4):
    axes[i].set_xlabel('Steps')

for i, gamma in enumerate(decays):
    ax = axes[i]
    ax.text(0.5, 1.1, r'$\gamma={:.3f}$'.format(gamma), transform=ax.transAxes,
            horizontalalignment='center', weight='bold')

pp.figtext(0.5, 0.90, u'Adaptation Decays More Quickly \N{RIGHTWARDS ARROW}',
           horizontalalignment='center', weight='bold')

fig.subplots_adjust(top=0.75, bottom=0.25)

pp.savefig('decay.pdf')
