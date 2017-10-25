import cPickle as pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as pp
import time

pp.ion()
pp.figure(figsize=(10, 10))

while True:
    pp.clf()

    accept = np.array(pickle.load(open('accept.pkl')))
    params = np.array(pickle.load(open('params.pkl')))
    derivs = np.array(pickle.load(open('derivs.pkl')))
    perms = np.array(pickle.load(open('perm.pkl')))

    pp.subplot(4, 2, 1)
    pp.plot(params[:, :, 0])
    pp.title('params0')

    pp.subplot(4, 2, 2)
    pp.plot(params[:, :, 1])
    pp.title('params1')

    pp.subplot(4, 2, 3)
    pp.plot(derivs[:, :, 0])
    pp.title('derivs0')

    pp.subplot(4, 2, 4)
    pp.plot(derivs[:, :, 1])
    pp.title('derivs1')

    pp.subplot(4, 2, 5)
    y = pd.rolling_mean((accept), 25)
    pp.plot(y)
    # pp.ylim(-0.05, 1.05)
    pp.title('mean accept')

    pp.subplot(4, 2, 6)
    # pp.plot(np.log10(np.product(accept, axis=1)))
    ln_A = np.log10(accept)
    ln_total_accept = np.sum(ln_A, axis=1)
    mean = np.mean(ln_A, axis=1)
    dev = (ln_A - mean[:, np.newaxis])**2
    sum_dev = np.sum(dev, axis=1)
    pp.plot(ln_total_accept - sum_dev)
    pp.title('log total accept')

    pp.subplot(4, 2, 7)
    current_index = np.array(range(perms.shape[1]))
    values = []
    for i in range(perms.shape[0]):
        new_value = np.zeros(perms.shape[1])
        new_value[perms[i, :]] = current_index
        values.append(new_value)
    pp.plot(np.array(values))

    pp.draw()
    pp.pause(0.0001)
    time.sleep(1)
