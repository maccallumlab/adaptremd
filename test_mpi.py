from __future__ import print_function
import numpy as np
from mpi4py import MPI

natoms = 5
comm = MPI.COMM_WORLD

data = np.zeros((2, natoms, 3))
my_data = np.zeros((natoms, 3))

if comm.rank == 0:
    data[0, :, :] = 0
    data[1, :, :] = 1

if comm.rank == 0:
    comm.Scatter(data, my_data, root=0)
else:
    comm.Scatter(None, my_data, root=0)

print('rank {}: {}'.format(comm.rank, my_data))
