import numpy as np
import remd
import restrainedprotein as rp
import adapt
import parallel_remd
from mpi4py import MPI

N = 10
TMIN = 300
TMAX = 700


if __name__ == '__main__':
    init_params = np.zeros((4, 2))
    init_params[:, 0] = np.linspace(TMIN, TMAX, 4)
    param_bounds = np.array([[TMIN, 0.0], [TMAX, 0.0]])


    results = np.zeros((N, N, 3))
    middle_values = np.linspace(TMIN, TMAX, N)
    for i, T1 in enumerate(middle_values):
        for j, T2 in enumerate(middle_values):
            init_params[1, 0] = T1
            init_params[2, 0] = T2
            r = remd.RemdLadder2(init_params)
            lr = adapt.LearningRateDecay(np.array((0, 0.0)), 0e-2)
            m = adapt.Adam2(0.9, 0.999, adapt.compute_derivative_log_total_acc, lr, param_bounds)

            variable_bonds = [[(3, 31, 0.0, 0.0, 0.0, 0.2)]]

            topname = 'ala.top'
            crdname = 'ala.crd'
            comm = MPI.COMM_WORLD

            parallel_remd.run(comm, 2000, r, m, update_every=5000, output_every=50, burn_in=1000, fixed_torsions=None,
                            fixed_bonds=None, variable_bonds=variable_bonds, topname=topname, crdname=crdname)

            if comm.rank == 0:
                results[i, j, :] = r.acceptance

    if comm.rank == 0:
        results = np.array(results)
        np.save('total_acc_test', results)
        print(results.shape)
