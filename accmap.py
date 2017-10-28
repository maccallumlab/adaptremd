import numpy as np
import remd
import restrainedprotein as rp
import adapt
import parallel_remd
from mpi4py import MPI


if __name__ == '__main__':
    init_params = np.zeros((3, 2))
    init_params[:, 0] = np.linspace(300, 450, 3)
    init_params[0, 0] = 300.
    param_bounds = np.array([[300.0, 0.0], [450.0, 0.0]])


    results = []
    middle_values = np.linspace(300, 450, 10)
    for mv in middle_values:
        init_params[1, 0] = mv
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
            results.append(r.acceptance)

    if comm.rank == 0:
        results = np.array(results)
        np.savetxt('total_acc_test.dat', results)
