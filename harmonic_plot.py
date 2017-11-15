import math
from matplotlib import pyplot as pp
import numpy as np
import eig


R = 8.314e-3
TMIN = 10.
TMAX = 10000.
N = 20
NSAMP = 10000

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
        [1.0 - 0.5 * a1,              0.5 * a1,                   0.0,            0.0],
        [      0.5 * a1, 1.0 - 0.5 * (a1 + a2),              0.5 * a2,            0.0],
        [           0.0,              0.5 * a2, 1.0 - 0.5 * (a2 + a3),       0.5 * a3],
        [           0.0,                   0.0,              0.5 * a3, 1.0 - 0.5 * a3]
    ])
    e1 = np.array([1.0, 0.0, 0.0])
    e1.shape = 1, 3
    e = np.array([1.0, 1.0, 1.0])
    e.shape = 3, 1

    Q = np.eye(3, 3) - P[:-1, :-1]
    x = np.linalg.inv(Q)
    rt = np.dot(e1, np.dot(x, e))
    rt = eig.fmpt(P)[0, -1]
    print rt
    return rt


def make_plot(temps, data):
    X, Y = np.meshgrid(temps, temps)
    xgrad, ygrad = np.gradient(data)
    pp.figure()
    pp.imshow(data.T, interpolation='nearest', origin='lower',
              extent=extent)
    pp.colorbar()

    # draw lines for expected results
    t2 = TMIN * (TMAX / TMIN)**(1.0 / 3.0)
    t3 = TMIN * (TMAX / TMIN)**(2.0 / 3.0)
    print t2, t3
    pp.axhline(t3, color='white', linewidth=3)
    pp.axvline(t2, color='white', linewidth=3)
    pp.axhline(t3, color='black', linewidth=1)
    pp.axvline(t2, color='black', linewidth=1)
    speed = (np.sqrt(xgrad**2 + ygrad**2))
    lw = 2 * np.sqrt(speed / speed.max())
    pp.streamplot(temps, temps, xgrad.T, ygrad.T, color='black', density=1.0,
                  linewidth=lw)


def make_plot2(temps, data):
    X, Y = np.meshgrid(temps, temps)
    pp.figure()
    pp.pcolor(X, Y, data.T)
    pp.colorbar()
    pp.xscale('log')
    pp.yscale('log')

    # draw lines for expected results
    t2 = TMIN * (TMAX / TMIN)**(1.0 / 3.0)
    t3 = TMIN * (TMAX / TMIN)**(2.0 / 3.0)
    pp.axhline(t3, color='white', linewidth=3)
    pp.axvline(t2, color='white', linewidth=3)
    pp.axhline(t3, color='black', linewidth=1)
    pp.axvline(t2, color='black', linewidth=1)


if __name__ == '__main__':
    round_trip_results = np.zeros((N, N))
    uniform_acc_results = np.zeros((N, N))
    total_acc_results = np.zeros((N, N))
    log_total_acc_results = np.zeros((N, N))

    temps = np.logspace(start=1, stop=4, num=N, base=10)
    for i, T2 in enumerate(temps):
        for j, T3 in enumerate(temps):
            a1 = compute_acc(TMIN, T2)
            a2 = compute_acc(T2, T3)
            a3 = compute_acc(T3, TMAX)

            ln1 = math.log(a1)
            ln2 = math.log(a2)
            ln3 = math.log(a3)
            mean = np.mean([a1, a2, a3])

            round_trip_results[i, j] = compute_rt(a1, a2, a3)
            uniform_acc_results[i, j] = -(a1 - mean)**2 - (a2 - mean)**2 - (a3 - mean)**2
            total_acc_results[i, j] = a1 * a2 * a3
            log_total_acc_results[i, j] = ln1 + ln2 + ln3


    make_plot2(temps, 1.0 / round_trip_results)
    make_plot2(temps, uniform_acc_results)
    make_plot2(temps, total_acc_results)
    make_plot2(temps, log_total_acc_results)
    pp.show()
