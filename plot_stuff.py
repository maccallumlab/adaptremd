import cPickle as pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as pp
import time

pp.ion()
pp.figure(figsize=(10, 10))

while True:
    pp.clf()

    try:
        accept = np.array(pickle.load(open('accept.pkl')))
        params = np.array(pickle.load(open('params.pkl')))
        derivs = np.array(pickle.load(open('derivs.pkl')))
        perms = np.array(pickle.load(open('perm.pkl')))
    except ValueError:
        pass
    n = params.shape[1]

    ax = pp.subplot(4, 2, 1)
    ax.set_color_cycle([pp.cm.viridis(c) for c in np.linspace(0, 1, n)])
    ax.plot(params[:, :, 0])
    ax.set_title('params0')

    ax = pp.subplot(4, 2, 2)
    ax.set_color_cycle([pp.cm.viridis(c) for c in np.linspace(0, 1, n)])
    ax.plot(params[:, :, 1])
    ax.set_title('params1')

    ax = pp.subplot(4, 2, 3)
    ax.set_color_cycle([pp.cm.viridis(c) for c in np.linspace(0, 1, n)])
    ax.plot(derivs[:, :, 0])
    ax.set_title('derivs0')

    ax = pp.subplot(4, 2, 4)
    ax.set_color_cycle([pp.cm.viridis(c) for c in np.linspace(0, 1, n)])
    ax.plot(derivs[:, :, 1])
    ax.set_title('derivs1')

    ax = pp.subplot(4, 2, 5)
    ax.set_color_cycle([pp.cm.viridis(c) for c in np.linspace(0, 1, n)])
    y = pd.rolling_mean((accept), 1)
    ax.plot(y)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('mean accept')

    ax = pp.subplot(4, 2, 6)
    ax.set_color_cycle([pp.cm.viridis(c) for c in np.linspace(0, 1, n)])
    ln_A = np.log10(accept)
    ln_total_accept = np.sum(ln_A, axis=1)
    mean = np.mean(ln_A, axis=1)
    dev = (ln_A - mean[:, np.newaxis])**2
    sum_dev = np.sum(dev, axis=1)
    ax.plot(ln_total_accept - sum_dev)
    ax.set_title('log total accept')

    ax = pp.subplot(4, 2, 7)
    ax.set_color_cycle([pp.cm.viridis(c) for c in np.linspace(0, 1, n)])
    current_index = np.array(range(perms.shape[1]))
    values = []
    for i in range(perms.shape[0]):
        new_value = np.zeros(perms.shape[1])
        new_value[perms[i, :]] = current_index
        values.append(new_value)
    ax.plot(np.array(values))

    ax = pp.subplot(4, 2, 8)
    ax.set_color_cycle([pp.cm.viridis(c) for c in np.linspace(0, 1, n)])
    s = pd.DataFrame(derivs[:, :, 0])
    ax.plot(s.expanding().mean())

    pp.draw()
    pp.pause(0.0001)
    time.sleep(1)
