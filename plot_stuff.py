import cPickle as pickle
import numpy as np
from matplotlib import pyplot as pp


accept = np.array(pickle.load(open('acceptance.pkl')))
params = np.array(pickle.load(open('params.pkl')))
derivs = np.array(pickle.load(open('derivs.pkl')))

pp.figure()
pp.plot(params[:, :, 0], marker='.')

print params.shape
for i in range(params.shape[1]):
    pp.axhline(params[0, i, 0], color='grey')
# factor = 1.1**0.33333
# pp.axhline(300.0 * factor, color='grey')
# pp.axhline(300.0 * factor**2, color='grey')



pp.figure()
pp.plot(derivs[:, :, 0], marker='.')

pp.figure()
pp.plot(accept, marker='.')

pp.show()


