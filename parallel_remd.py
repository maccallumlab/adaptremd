from __future__ import print_function
from mpi4py import MPI
import sys
from collections import namedtuple, defaultdict
import platform
import os
import cPickle as pickle
import numpy as np
import remd
import restrainedprotein as rp
import adapt
import time


#
# setup exception handling to abort
#
sys_excepthook = sys.excepthook

def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    MPI.COMM_WORLD.Abort(1)

sys.excepthook = mpi_excepthook


def run_master(comm, adapt_iter, r, l, fixed_torsions,
               fixed_bonds, variable_bonds, topname, crdname):
    n_replicas = r.n_walkers
    print('Starting replica exchange')
    dev_id = negotiate_device_id(comm)

    # create walker on device
    w = rp.RestrainedProtein(comm.rank, topname, crdname,
                             300.0, np.array([0.0] * len(variable_bonds)),
                             variable_bonds, fixed_bonds, fixed_torsions,
                             nsteps=500, output_steps=500*adapt_iter.initial_cycle_length,
                             gpuid=dev_id)
    # setup states
    states = [w.x] * n_replicas

    # setup history
    param_history = []
    param_history.append(r.params.copy())
    accept_history = []

    # setup walker history
    walker_perm = list(range(n_replicas))
    perm_hist = []

    for adapt_step in adapt_iter:
        # update our parameters
        params = []
        for i in range(n_replicas):
            params.append(r.params[i, :])

        print('Running {} of {}'.format(adapt_step.step, adapt_iter.max_steps))

        # get state and parameters from master
        package = [(s, p) for s, p in zip(states, params)]
        s, p = scatter_state_params(comm, package)
        w.x = s
        w.params = p

        # simulate
        w.update()

        # get states from slaves
        states = gather_state(comm, w.x)

        # get list of states and parameters from master
        test_states = broadcast_states_for_energy_deriv(comm, states)

        p = scatter_params_for_energy_deriv(comm, params)

        # compute energies and derivs
        energies = []
        derivs = []
        for s in test_states:
            energies.append(w.get_trial_energy(s))
            derivs.append(w.get_trial_derivs(s))

        # gather energies and derivs
        package = gather_energy_deriv(comm, (energies, derivs))
        energy_matrix = np.array([p[0] for p in package])
        deriv_matrix = np.array([p[1] for p in package])

        # do remd
        perm = r.update(energy_matrix, deriv_matrix)
        new_states = []
        new_walker_perm = []
        for p in perm:
            new_states.append(states[p])
            new_walker_perm.append(walker_perm[p])
        states = new_states
        walker_perm = new_walker_perm
        perm_hist.append(walker_perm)

        # reset after burn-in period
        if adapt_step.burn_in_only:
            print('Step {}, burning in.'.format(adapt_step.step))
            r.reset_stats()

        # do update if it's time
        if adapt_step.update:
            l.update(r, adapt_step.learning_rate)

        # do output if it's time
        if adapt_step.update or adapt_step.output_only:
            param_history.append(r.params.copy())
            with open('params.pkl', 'w') as outfile:
                pickle.dump(param_history, outfile)

            accept_history.append(r._acc / r.n_trials)
            with open('accept.pkl', 'w') as outfile:
                pickle.dump(accept_history, outfile)

            with open('derivs.pkl', 'w') as outfile:
                pickle.dump(l.derivs, outfile)

            with open('perm.pkl', 'w') as outfile:
                pickle.dump(perm_hist, outfile)

        # reset stats if we did output
        if adapt_step.update:
            r.reset_stats()


def run_slave(comm, adapt_iter, fixed_torsions, fixed_bonds, variable_bonds, topname, crdname):
    dev_id = negotiate_device_id(comm)

    # create walker on device
    w = rp.RestrainedProtein(comm.rank, topname, crdname,
                             300.0, np.array([0.0] * len(variable_bonds)),
                             variable_bonds, fixed_bonds, fixed_torsions, nsteps=500,
                             output_steps=500*adapt_iter.initial_cycle_length,
                             gpuid=dev_id)

    for adapt_step in adapt_iter:
        # get pos, vel, and parameters from master
        s, p = scatter_state_params(comm)
        w.x = s
        w.params = p

        # simulate
        w.update()

        # send state back to master
        gather_state(comm, w.x)

        # get list of states and parameters from master
        test_states = broadcast_states_for_energy_deriv(comm)
        p = scatter_params_for_energy_deriv(comm)

        # compute energies and derivs
        energies = []
        derivs = []
        for s in test_states:
            energies.append(w.get_trial_energy(s))
            derivs.append(w.get_trial_derivs(s))

        # send energies and derivs back to master
        gather_energy_deriv(comm, (energies, derivs))


#
# MPI Stuff
#
def scatter_state_params(comm, package=None):
    return comm.scatter(package, root=0)


def gather_state(comm, state):
    return comm.gather(state, root=0)


def broadcast_states_for_energy_deriv(comm, package=None):
    return comm.bcast(package, root=0)

def scatter_params_for_energy_deriv(comm, package=None):
    return comm.scatter(package, root=0)

def gather_energy_deriv(comm, package):
    return comm.gather(package, root=0)

def negotiate_device_id(comm):
    hostname = platform.node()
    try:
        visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
        visible_devices = visible_devices.split(',')
        if visible_devices:
            visible_devices = [int(x) for x in visible_devices]
        else:
            raise RuntimeError('No cuda devices available')
    except KeyError:
        print('CUDA_VISIBLE_DEVICES is not set')
        visible_devices = None

    hosts = comm.gather((hostname, visible_devices), root=0)

    # The master computes device ids
    if comm.rank == 0:
        # If CUDA_VISIBLE_DEVICES is not set on the master, it
        # shouldn't be set anywhere else. We'll number the devices
        # starting from zero.
        if hosts[0][1] is None:
            host_counts = defaultdict(int)
            device_ids = []
            for host in hosts:
                assert host[1] is None
                device_ids.append(host_counts[host[0]])
                host_counts[host[0]] += 0
        else:
            available_devices = {}
            for host in hosts:
                if host[0] in available_devices:
                    if host[1] != available_devices[host[0]]:
                        raise RuntimeError('GPU devices for host do not match')
                else:
                    available_devices[host[0]] = host[1]

            for host in hosts:
                # CUDA numbers devices from zero always, so even if
                # CUDA_VISIBLE_DEVICES=2,3 we would need to ask for 0 or 1.
                # So, we subtract the minimum value, except we leave -1.
                min_device_id = min(available_devices[host[0]])
                if min_device_id != -1:
                    available_devices[host[0]] = [
                        device_id - min_device_id for device_id in
                        available_devices[host[0]]
                    ]

            device_ids = []
            for host in hosts:
                dev_id = available_devices[host[0]][0]
                if dev_id == -1:
                    device_ids.append(-1)
                else:
                    device_ids.append(available_devices[host[0]].pop(0))

    # the slaves do nothing
    else:
        device_ids = None

    # communicate device ids from master to slaves
    device_id = comm.scatter(device_ids, root=0)

    return device_id


def run(comm, adapt_iter, remd, learn, fixed_torsions=None, fixed_bonds=None,
        variable_bonds=None, topname='topol.top', crdname='inpcrd.crd'):
    rank = comm.Get_rank()

    fixed_torsions = fixed_torsions if fixed_torsions else []
    fixed_bonds = fixed_bonds if fixed_bonds else []
    variable_bonds = variable_bonds if variable_bonds else []

    if rank == 0:
        run_master(comm, adapt_iter, remd, learn, fixed_torsions,
                   fixed_bonds, variable_bonds, topname, crdname)
        print(remd.acceptance)
    else:
        run_slave(comm, adapt_iter, fixed_torsions, fixed_bonds,
                  variable_bonds, topname, crdname)


if __name__ == '__main__':
    init_params = np.zeros((8, 2))
    init_params[:, 0] = np.linspace(440, 450, 8)
    init_params[0, 0] = 300.0
    param_bounds = np.array([[300.0, 0.0], [450.0, 100.0]])
    # init_params[:, 0] = 300.0
    # init_params[:, 1] = np.linspace(500.0, 0.0, 8)
    r = remd.RemdLadderJensen2(init_params)
    # lr = adapt.LearningRateDecay(np.array((1e-3, 0.0)), 1e-2)
    # m = adapt.MomentumSGD2(0.9, adapt.compute_derivative_jensen_pen, lr, param_bounds)
    lr = adapt.LearningRateDecay(np.array((2, 0.0)), 1e-2)
    m = adapt.Adam2(0.9, 0.999, adapt.compute_derivative_jensen_pen, param_bounds)
    adapt_iter = adapt.AdaptationIter(max_steps=100000,
                                      discard_first_steps=10,
                                      init_cycle_length=16,
                                      cycle_length_doubling_cycles=50,
                                      fraction_batch_discard=0.5,
                                      learning_rate_decay_cycles=50,
                                      init_learning_rate=numpy.array([1, 2]))


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

    run(adapt_iter, r, m, fixed_torsions=None, fixed_bonds=None,
        variable_bonds=variable_bonds)
