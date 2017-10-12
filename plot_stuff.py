import cPickle as pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as pp


accept = np.array(pickle.load(open('accept.pkl')))
params = np.array(pickle.load(open('params.pkl')))
derivs = np.array(pickle.load(open('derivs.pkl')))

pp.figure()
pp.plot(params[:, :, 0], marker='.')

pp.figure()
pp.plot(params[:, :, 1], marker='.')

pp.figure()
y = pd.rolling_mean(accept, 25)
print y.shape
pp.plot(y, marker='.')
pp.axhline(0.0, color='black')
pp.ylim(-0.05, 1.05)

pp.figure()
pp.plot(derivs[:, :, 0], marker='.')
pp.axhline(0.0, color='black')

pp.figure()
pp.plot(derivs[:, :, 1], marker='.')
pp.axhline(0.0, color='black')

pp.figure()
pp.plot(np.log10(np.product(accept, axis=1)))

pp.show()
