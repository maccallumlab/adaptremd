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

    pp.subplot(3, 2, 1)
    pp.plot(params[:, :, 0])
    pp.title('params0')

    pp.subplot(3, 2, 2)
    pp.plot(params[:, :, 1])
    pp.title('params1')

    pp.subplot(3, 2, 3)
    pp.plot(derivs[:, :, 0])
    pp.title('derivs0')

    pp.subplot(3, 2, 4)
    pp.plot(derivs[:, :, 1])
    pp.title('derivs1')

    pp.subplot(3, 2, 5)
    y = pd.rolling_mean(accept, 5)
    print y.shape
    pp.plot(y)
    pp.ylim(-0.05, 1.05)
    pp.title('mean accept')

    pp.subplot(3, 2, 6)
    pp.plot(np.log10(np.product(accept, axis=1)))
    pp.title('log total accept')

    pp.draw()
    pp.pause(0.0001)
    time.sleep(5)
