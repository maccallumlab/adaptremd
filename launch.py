from __future__ import print_function
import numpy as np
import remd
import restrainedprotein as rp
import adapt
import parallel_remd
from mpi4py import MPI


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    N = 12
    init_params = np.zeros((N, 2))
    # init_params[:, 0] = np.linspace(300, 400, N)
    # init_params[-1, 0] = 700.
    init_params[:, 0] = np.linspace(300., 450, N)
    init_params[:, 1] = np.linspace(5.52, -10, N)
    param_bounds = np.array([[300.0, -10.], [450.0, 5.52]])
    r = remd.RemdLadder2(init_params)
    lr = adapt.LearningRateDecay(np.array((1., 0.1)), 0.0)
    m = adapt.Adam2(0.9, 0.9, adapt.compute_derivative_log_total_acc, param_bounds)

    torsions = []
    with open('torsions.dat') as infile:
        for line in infile:
            i, j, k, l, theta0, delta = line.split()
            i, j, k, l = [int(x) - 1 for x in (i, j, k, l)]
            theta0 = float(theta0)
            delta = float(delta)
            torsions.append((l, k, j, i, theta0, delta))

    fixed_bonds = []
    with open('fixed_bonds.dat') as infile:
        for line in infile:
            i, j, d = line.split()
            i = int(i)
            j = int(j)
            d = float(d)
            d1 = max(0.0, d-0.1)
            d2 = d
            d3 = d
            d4 = d + 0.1
            fixed_bonds.append((i, j, d1, d2, d3, d4))

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

    # topname = 'ala.top'
    # crdname = 'ala.crd'
    topname = 'topol.top'
    crdname = 'inpcrd.crd'

    adapt_iter = adapt.AdaptationIter(max_steps=100000,
                                      discard_first_steps=1000,
                                      init_cycle_length=16,
                                      cycle_length_doubling_cycles=50,
                                      fraction_batch_discard=0.5,
                                      learning_rate_decay_cycles=50,
                                      init_learning_rate=np.array([4, 0.2]))

    parallel_remd.run(comm, adapt_iter, r, m, fixed_torsions=torsions,
                      fixed_bonds=fixed_bonds, variable_bonds=variable_bonds,
                      topname=topname, crdname=crdname)
