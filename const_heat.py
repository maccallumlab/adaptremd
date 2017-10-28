import math
from matplotlib import pyplot as pp
import numpy as np


R = 8.314e-3
Cv = 10.
T1 = 300.
T4 = 330.
TMIN = T1
TMAX = T4
N = 20
NSAMP = 100000

def compute_acc(T1, T2):
    mean1 = Cv * T1
    var1 = math.sqrt(R * T1**2 * Cv)
    mean2 = Cv * T2
    var2 = math.sqrt(R * T2**2 * Cv)
    E1 = np.random.normal(mean1, var1, NSAMP)
    E2 = np.random.normal(mean2, var2, NSAMP)

    delta_beta = 1.0 / R / T2 - 1.0 / R / T1
    delta_E = E2 - E1
    acc = np.minimum(1.0, np.exp(delta_beta * delta_E))
    return np.mean(acc)

res1 = np.zeros((N, N))
res2 = np.zeros((N, N))
res3 = np.zeros((N, N))

temps = np.linspace(TMIN, TMAX, N)
for i, T2 in enumerate(temps):
    for j, T3 in enumerate(temps):
        a1 = compute_acc(T1, T2)
        a2 = compute_acc(T2, T3)
        a3 = compute_acc(T3, T4)

        ln1 = math.log(a1)
        ln2 = math.log(a2)
        ln3 = math.log(a3)
        mean = np.mean([ln1, ln2, ln3])

        res1[i, j] = ln1 + ln2 + ln3
        res2[i, j] = -(ln1 - mean)**2 - (ln2 - mean)**2 - (ln3 - mean)**2
        res3[i, j] = a1 * a2 * a3

X, Y = np.meshgrid(temps, temps)

xgrad, ygrad = np.gradient(res1)
pp.figure()
pp.imshow(res1[::-1, ::-1], interpolation='nearest', origin='lower', extent=[TMIN, TMAX, TMIN, TMAX])
pp.colorbar()
pp.quiver(Y, X, xgrad, ygrad)

pp.figure()
pp.imshow(res2[::-1, ::-1], interpolation='nearest', origin='lower', extent=[TMIN, TMAX, TMIN, TMAX])
pp.colorbar()
xgrad2, ygrad2 = np.gradient(res2)
pp.quiver(Y, X, xgrad2, ygrad2)

pp.figure()
pp.imshow(res3[::-1, ::-1], interpolation='nearest', origin='lower', extent=[TMIN, TMAX, TMIN, TMAX])
pp.colorbar()
xgrad3, ygrad3 = np.gradient(res3)
pp.quiver(Y, X, xgrad3, ygrad3)

res4 = res1 + 10 * res3
pp.figure()
pp.imshow(res4[::-1, ::-1], interpolation='nearest', origin='lower', extent=[TMIN, TMAX, TMIN, TMAX])
pp.colorbar()
xgrad4, ygrad4 = np.gradient(res4)
pp.quiver(Y, X, xgrad4, ygrad4)

res5 = res1 + res2
pp.figure()
pp.imshow(res5[::-1, ::-1], interpolation='nearest', origin='lower', extent=[TMIN, TMAX, TMIN, TMAX])
pp.colorbar()
xgrad5, ygrad5 = np.gradient(res5)
pp.quiver(Y, X, xgrad5, ygrad5)


pp.show()
