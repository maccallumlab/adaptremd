from __future__ import print_function
import numpy as np
import remd
import restrainedprotein as rp
import adapt
import parallel_remd
from mpi4py import MPI


if __name__ == '__main__':
    N = 8
    init_params = np.zeros((N, 2))
    init_params[:, 0] = np.linspace(345, 350, N)
    init_params[0, 0] = 300.
    param_bounds = np.array([[300.0, 0.0], [350.0, 0.0]])
    # init_params[:, 0] = 300.0
    # init_params[:, 1] = np.linspace(500.0, 0.0, 8)
    r = remd.RemdLadder2(init_params)
    # lr = adapt.LearningRateDecay(np.array((2.0, 0.0)), 0)
    # m = adapt.MomentumSGD2(0.9, adapt.compute_derivative_log_total_acc, lr, param_bounds)
    lr = adapt.LearningRateDecay(np.array((2, 0.0)), 0e-2)
    m = adapt.Adam2(0.9, 0.999, adapt.compute_derivative_log_total_acc, lr, param_bounds)

    torsions = []
    with open('torsions.dat') as infile:
        for line in infile:
            i, j, k, l, theta0, delta = line.split()
            i, j, k, l = [int(x) - 1 for x in (i, j, k, l)]
            theta0 = float(theta0)
            delta = float(delta)
            torsions.append((i, j, k, l, theta0, delta))

    fixed_bonds = []

    variable_bonds = []
    for i in range(1):
        bond_list = []
        with open('variable_bonds{}.dat'.format(i)) as infile:
            for line in infile:
                i, j, d = line.split()
                i = int(i) - 1
                j = int(j) - 1
                d = float(d)
                d1 = max(0.0, d - 0.1)
                d2 = d
                d3 = d
                d4 = d + 0.1
                bond_list.append((i, j, d1, d2, d3, d4))
        variable_bonds.append(bond_list)
    variable_bonds = [[(3, 31, 0.0, 0.0, 0.0, 0.2)]]

    topname = 'topol.top'
    crdname = 'inpcrd.crd'
    # topname = 'topol.top'
    # crdname = 'inpcrd.crd'

    comm = MPI.COMM_WORLD
    parallel_remd.run(comm, 100000, r, m, update_every=10, output_every=50, burn_in=10, fixed_torsions=None,
                      fixed_bonds=None, variable_bonds=variable_bonds, topname=topname, crdname=crdname)
