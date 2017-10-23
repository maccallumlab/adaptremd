#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as pp
from simplelandscape import (
    flat_energy, linear_energy, logistic_energy, quadratic_energy,
    umbrella_bias, OneDimLandscape)
from remd import RemdLadder, RemdLadderJensen
import adapt


n_cond = 8
bias_x = np.linspace(550, 600, n_cond)
bias_x[0] = 400

momentums = [0.999, 0.99, 0.9, 0.0]
learning_rates = list(reversed([10.0, 1.0, 0.1, 0.01]))

plot_index = 1

fig, axes = pp.subplots(len(momentums), len(learning_rates),
                        sharex=True, sharey=True, figsize=(6, 6))

for i, momentum in enumerate(momentums):
    for j, learning_rate in enumerate(learning_rates):
        walkers = []
        for x in bias_x:
            p = np.array((x, 0.025))
            w = OneDimLandscape(logistic_energy, umbrella_bias, x, p)
            walkers.append(w)

        r = RemdLadderJensen(walkers)

        lr = adapt.LearningRateDecay(np.array((learning_rate, 0)), 0)
        param_bounds = np.array([[0, 0.025], [1000, 0.025]], dtype=float)
        m = adapt.MomentumSGD(momentum, adapt.compute_derivative_jensen,
                              lr, param_bounds)
        # lr = adapt.LearningRateDecay(np.array((20, 1e-3)), 1e-2)
        # m = adapt.Adam(0.99, 0.999, adapt.compute_derivative_log_total_acc, lr)
        a = adapt.Adaptor(r, 8, m)

        a.run(2000)

        params = np.array(a.params)
        axes[i, j].set_color_cycle([pp.cm.viridis(c) for c in np.linspace(0, 1, 8)])

        if np.logical_or(params[:, :, 0] < 200, params[:, :, 0] > 800).any():
            axes[i, j].text(0.5, 0.5, 'unstable', transform=axes[i, j].transAxes,
                            horizontalalignment='center', verticalalignment='center')
        else:
            axes[i, j].plot(params[:, :, 0])

        axes[i, j].set_ylim(370, 630)
        axes[i, j].xaxis.set_major_locator(pp.FixedLocator([0, 1000, 2000]))


        plot_index += 1

# set the labels
for i in range(4):
    axes[i, 0].set_ylabel('$x_0$') 
    axes[3, i].set_xlabel('Steps')

for i, lr in enumerate(learning_rates):
    ax = axes[0, i]
    ax.text(0.5, 1.2, r'$\alpha={}$'.format(lr), transform=ax.transAxes,
            horizontalalignment='center', weight='bold')

for i, mom in enumerate(momentums):
    ax = axes[i, 3]
    ax.text(1.1, 0.5, r'$\mu={}$'.format(mom), transform=ax.transAxes,
            verticalalignment='center', weight='bold',
            rotation='vertical')


pp.figtext(0.5, 0.95, u'Increasing Learning Rate \N{RIGHTWARDS ARROW}',
           horizontalalignment='center', weight='bold')

pp.figtext(0.95, 0.5, u'Increasing Momentum \N{RIGHTWARDS ARROW}',
           verticalalignment='center', weight='bold', rotation='vertical')

pp.savefig('momentum_learning_rate.pdf')
