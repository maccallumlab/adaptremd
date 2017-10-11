import cPickle as pickle
import numpy as np
from matplotlib import pyplot as pp


accept = np.array(pickle.load(open('accept.pkl')))
params = np.array(pickle.load(open('params.pkl')))
derivs = np.array(pickle.load(open('derivs.pkl')))

pp.figure()
pp.plot(params[:, :, 0], marker='.')

pp.figure()
pp.plot(accept, marker='.')
pp.ylim(-0.05, 1.05)

pp.figure()
pp.plot(derivs[:, :, 0], marker='.')
pp.axhline(0.0, color='black')

pp.show()
