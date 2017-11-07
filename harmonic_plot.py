import math
from matplotlib import pyplot as pp
import numpy as np


R = 8.314e-3
TMIN = 10.
TMAX = 10000.
N = 100
NSAMP = 100000

d = (TMAX - TMIN) / float(N)
extent = (TMIN-0.5*d, TMAX+0.5*d, TMIN-0.5*d, TMAX+0.5*d)


def compute_acc(T1, T2):
    std1 = math.sqrt(R * T1 / 2.0)
    std2 = math.sqrt(R * T2 / 2.0)

    x1 = np.random.normal(0, std1, NSAMP)
    x2 = np.random.normal(0, std2, NSAMP)

    E11 = x1**2 / R / T1
    E22 = x2**2 / R / T2
    E12 = x1**2 / R / T2
    E21 = x2**2 / R / T1

    delta = E11 + E22 - E12 - E21
    acc = np.minimum(1.0, np.exp(delta))
    return np.mean(acc)


def compute_rt(a1, a2, a3):
    P = np.array([
        [1.0 - a1,                          a1,              0.0,                 0.0],
        [0.5 * a1, (1.0 - 0.5 * a1 - 0.5 * a2),              0.5 * a2,            0.0],
        [     0.0,                    0.5 * a2, (1.0 - 0.5 * a2 - 0.5 * a3), 0.5 * a3],
        [     0.0,                         0.0,                          a3, 1.0 - a3]
    ])
    e1 = np.array([1.0, 0.0, 0.0])
    e1.shape = 1, 3
    e = np.array([1.0, 1.0, 1.0])
    e.shape = 3, 1

    Q = np.eye(3, 3) - P[:-1, :-1]
    x = np.linalg.lstsq(Q, e)[0]
    rt = np.dot(e1, x)
    return rt


def make_plot(temps, data):
    X, Y = np.meshgrid(temps, temps)
    xgrad, ygrad = np.gradient(data)
    pp.figure()
    pp.imshow(data.T, interpolation='nearest', origin='lower',
              extent=extent)
    pp.colorbar()
    # pp.quiver(Y, X, xgrad, ygrad, pivot='mid')
    speed = (np.sqrt(xgrad**2 + ygrad**2))
    lw = 2 * np.sqrt(speed / speed.max())
    pp.streamplot(temps, temps, xgrad.T, ygrad.T, color='black', density=1.0,
                  linewidth=lw)


if __name__ == '__main__':
    res1 = np.zeros((N, N))
    res2 = np.zeros((N, N))
    res3 = np.zeros((N, N))
    res4 = np.zeros((N, N))
    res5 = np.zeros((N, N))
    res6 = np.zeros((N, N))
    res7 = np.zeros((N, N))

    temps = np.linspace(TMIN, TMAX, N)
    for i, T2 in enumerate(temps):
        for j, T3 in enumerate(temps):
            a1 = compute_acc(TMIN, T2)
            a2 = compute_acc(T2, T3)
            a3 = compute_acc(T3, TMAX)

            ln1 = math.log(a1)
            ln2 = math.log(a2)
            ln3 = math.log(a3)
            mean = np.mean([ln1, ln2, ln3])
            # mean = math.log(0.90)

            res1[i, j] = ln1 + ln2 + ln3
            res2[i, j] = -(ln1 - mean)**2 - (ln2 - mean)**2 - (ln3 - mean)**2
            res3[i, j] = a1 * a2 * a3
            res4[i, j] = a1
            res5[i, j] = a2
            res6[i, j] = a3
            res7[i, j] = compute_rt(a1, a2, a3)


    make_plot(temps, res1)
    # make_plot(temps, res2)
    make_plot(temps, res3)
    # make_plot(temps, res1 + res2)
    make_plot(temps, res4)
    make_plot(temps, res5)
    make_plot(temps, res6)
    make_plot(temps, -np.log(res7))
    pp.show()
