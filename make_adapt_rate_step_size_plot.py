#!/usr/bin/env python

from __future__ import division
import math
import numpy as np
from matplotlib import pyplot as pp
from harmonic import HarmonicLandscape
from remd import RemdLadder
import adapt


n_cond = 4
init_temps = np.array([10, 5000, 5000, 10000], dtype=float)

T2 = 10 * (10000 / 10) ** (1.0 / 3.0)
T3 = 10 * (10000 / 10) ** (2.0 / 3.0)

adapt_steps = [4, 16, 64, 256]
learning_rates = [25.0, 50.0, 100.0, 200.0]

plot_index = 1

fig, axes = pp.subplots(len(adapt_steps), len(learning_rates),
                        sharex=True, sharey=True, figsize=(6, 6))

for i, adapt_step in enumerate(adapt_steps):
    for j, learning_rate in enumerate(learning_rates):
        walkers = []
        for t in init_temps:
            w = HarmonicLandscape(t)
            walkers.append(w)
        r = RemdLadder(walkers)

        lr = adapt.LearningRateDecay(np.array((learning_rate,)), 0)
        param_bounds = np.array([[10], [10000]], dtype=float)
        m = adapt.Adam(0.9, 0.9, adapt.compute_derivative_log_total_acc,
                       lr, param_bounds)
        a = adapt.Adaptor(r, adapt_step, m)

        n_steps = 100000 // adapt_step
        a.run(n_steps)

        params = np.array(a.params)
        # axes[i, j].set_color_cycle([pp.cm.viridis(c) for c in np.linspace(0, 1, n_cond)])

        # if np.logical_or(params[:, :, 0] < 200, params[:, :, 0] > 800).any():
        #     axes[i, j].text(0.5, 0.5, 'unstable', transform=axes[i, j].transAxes,
        #                     horizontalalignment='center', verticalalignment='center')
        # else:
        x = adapt_step * np.linspace(0, n_steps, params.shape[0])
        axes[i, j].plot(x, params[:, 1:3, 0], linewidth=2)
        axes[i, j].axhline(T2, color='white', linewidth=2)
        axes[i, j].axhline(T2, color='black', linewidth=1, linestyle='--')
        axes[i, j].axhline(T3, color='white', linewidth=2)
        axes[i, j].axhline(T3, color='black', linewidth=1, linestyle='--')

        axes[i, j].set_ylim(-1000, 5000)
        axes[i, j].set_xlim(0, 100000)
        axes[i, j].yaxis.set_major_locator(pp.FixedLocator([0, 5000, 10000]))
        axes[i, j].xaxis.set_major_locator(pp.FixedLocator([0, 100000]))
        axes[i, j].xaxis.set_major_formatter(pp.FixedFormatter(['0', '$10^5$']))


        plot_index += 1

# set the labels
for i in range(4):
    axes[i, 0].set_ylabel('Temp. (K)')
    axes[3, i].set_xlabel('Steps')

for i, lr in enumerate(learning_rates):
    ax = axes[0, i]
    ax.text(0.5, 1.2, r'$\alpha={}$'.format(lr), transform=ax.transAxes,
            horizontalalignment='center', weight='bold')

for i, adapt_step in enumerate(adapt_steps):
    ax = axes[i, 3]
    ax.text(1.1, 0.5, r'$\eta={}$'.format(adapt_step), transform=ax.transAxes,
            verticalalignment='center', weight='bold',
            rotation='vertical')

color_cycle = pp.rcParams['axes.prop_cycle'].by_key()['color']
axes[0, 0].text(7.5e4, 3000, 'T2', color=color_cycle[0], weight='bold')
axes[0, 0].text(7.5e4, 4000, 'T3', color=color_cycle[1], weight='bold')

pp.figtext(0.5, 0.95, u'Increasing Learning Rate \N{RIGHTWARDS ARROW}',
           horizontalalignment='center', weight='bold')

pp.figtext(0.95, 0.5, u'More Frequent Adaptation \N{RIGHTWARDS ARROW}',
           verticalalignment='center', weight='bold', rotation='vertical')

pp.savefig('adapt_rate_step_size.pdf')
