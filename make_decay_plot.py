#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as pp
from simplelandscape import (
    flat_energy, linear_energy, logistic_energy, quadratic_energy,
    umbrella_bias, OneDimLandscape)
from remd import RemdLadder
import adapt


n_cond = 8
bias_x = np.linspace(550, 600, n_cond)
bias_x[0] = 400

decays = [1e-3, 1e-2, 1e-1, 1.0]
fig, axes = pp.subplots(1, len(decays), sharex=True, sharey=True, figsize=(6, 2))

for i, decay in enumerate(decays):
    walkers = []
    for x in bias_x:
        p = np.array((x, 0.025))
        w = OneDimLandscape(logistic_energy, umbrella_bias, x, p)
        walkers.append(w)

    r = RemdLadder(walkers)

    lr = adapt.LearningRateDecay(np.array((1.0, 0)), decay)
    m = adapt.MomentumSGD(0.9, adapt.compute_derivative_log_total_acc, lr)
    a = adapt.Adaptor(r, 8, m)

    a.run(2000)

    params = np.array(a.params)
    axes[i].set_color_cycle([pp.cm.viridis(c) for c in np.linspace(0, 1, 8)])
    axes[i].plot(params[:, :, 0])
    axes[i].set_ylim(370, 630)
    axes[i].xaxis.set_major_locator(pp.FixedLocator([0, 1000, 2000]))

# set the labels
axes[0].set_ylabel('$x_0$')
for i in range(4):
    axes[i].set_xlabel('Steps')

for i, decay in enumerate(decays):
    ax = axes[i]
    ax.text(0.5, 1.1, r'$\gamma={}$'.format(decay), transform=ax.transAxes,
            horizontalalignment='center', weight='bold')

pp.figtext(0.5, 0.90, u'Increasing Learning Rate Decay \N{RIGHTWARDS ARROW}',
           horizontalalignment='center', weight='bold')

fig.subplots_adjust(top=0.75, bottom=0.25)

pp.savefig('decay.pdf')
