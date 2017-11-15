#!/usr/bin/env python

from __future__ import division
import math
import numpy as np
from matplotlib import pyplot as pp
import matplotlib.patheffects as PathEffects
from harmonic import HarmonicLandscape
from remd import RemdLadder
import adapt


n_cond = 4
init_temps = np.array([10, 5000, 5000, 10000], dtype=float)

T2 = 10 * (10000 / 10) ** (1.0 / 3.0)
T3 = 10 * (10000 / 10) ** (2.0 / 3.0)

gamma1s = [0, 1e-3, 5e-3, 25e-3]
gamma2s = [1.0] + [2.0 ** (1.0 / n) for n in [320, 80, 20]]

plot_index = 1

fig, axes = pp.subplots(len(gamma1s), len(gamma2s),
                        sharex=True, sharey=True, figsize=(6, 6))

for i, gamma1 in enumerate(gamma1s):
    for j, gamma2 in enumerate(gamma2s):
        walkers = []
        for t in init_temps:
            w = HarmonicLandscape(t)
            walkers.append(w)
        r = RemdLadder(walkers)

        lr = adapt.LearningRateDecay(np.array((100,)), gamma1)
        param_bounds = np.array([[10], [10000]], dtype=float)
        m = adapt.Adam(0.9, 0.9, adapt.compute_derivative_log_total_acc,
                       lr, param_bounds)
        a = adapt.ExponentialAdaptor(r, 4, gamma2, m)

        target_iters = 100000
        iterations = a.min_iterations_for_target_steps(target_iters)
        print iterations
        a.run(iterations)

        params = np.array(a.params)
        axes[j, i].plot(a.adapted_at, params[:, 1:3, 0], linewidth=2)
        axes[j, i].axhline(T2, color='white', linewidth=2)
        axes[j, i].axhline(T2, color='black', linewidth=1, linestyle='--')
        axes[j, i].axhline(T3, color='white', linewidth=2)
        axes[j, i].axhline(T3, color='black', linewidth=1, linestyle='--')

        axes[j, i].set_ylim(-1000, 5000)
        axes[j, i].set_xlim(0, target_iters)
        axes[j, i].yaxis.set_major_locator(pp.FixedLocator([0, 5000, 10000]))
        axes[j, i].xaxis.set_major_locator(pp.FixedLocator([0, 100000]))
        axes[j, i].xaxis.set_major_formatter(pp.FixedFormatter(['0', '$10^5$']))

        plot_index += 1

# set the labels
for i in range(4):
    axes[i, 0].set_ylabel('Temp. (K)')
    axes[3, i].set_xlabel('Steps')

for i, value in enumerate(gamma1s):
    ax = axes[0, i]
    ax.text(0.5, 1.2, r'$\gamma_1={}\times 10^{{-3}}$'.format(int(value * 1000)), transform=ax.transAxes,
            horizontalalignment='center', weight='bold')

for i, value in enumerate(gamma2s):
    ax = axes[i, 3]
    ax.text(1.1, 0.5, r'$\gamma_2={:.3f}$'.format(value), transform=ax.transAxes,
            verticalalignment='center', weight='bold',
            rotation='vertical')

color_cycle = pp.rcParams['axes.prop_cycle'].by_key()['color']
txt = axes[0, 0].text(7.5e4, 3000, 'T2', color=color_cycle[0], weight='bold')
txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground='w'), PathEffects.Normal()])
txt = axes[0, 0].text(7.5e4, 4000, 'T3', color=color_cycle[1], weight='bold')
txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground='w'), PathEffects.Normal()])

pp.figtext(0.5, 0.95, u'Faster Learning Rate Decay \N{RIGHTWARDS ARROW}',
           horizontalalignment='center', weight='bold')

pp.figtext(0.95, 0.5, u'\N{LEFTWARDS ARROW} Faster Adaptation Rate Decay',
           verticalalignment='center', weight='bold', rotation='vertical')

pp.savefig('both_decay.pdf')
