from __future__ import print_function
from simtk import openmm as mm
from simtk.openmm import app
import simtk.unit as u
import numpy as np
import collections
import math


# ideal gas constant
R = 8.314e-3


Torsion = collections.namedtuple('Torsion', 'i j k l angle delta_angle force_const')

Spring = collections.namedtuple('Spring', 'i j r1 r2 r3 r4 force_const')


class RestrainedProtein(object):
    def __init__(self, output_index, parm_path, crd_path, init_temp, init_k,
                 variable_springs, fixed_springs, fixed_torsions, nsteps=500):
        self.output_index = output_index
        self.nsteps = nsteps
        self.temperature = init_temp
        self.k = init_k

        prmtop = app.AmberPrmtopFile(parm_path)
        inpcrd = app.AmberInpcrdFile(crd_path)

        self.system = prmtop.createSystem(nonbondedMethod=app.CutoffNonPeriodic,
                                          nonbondedCutoff=1.8*u.nanometer,
                                          constraints=app.HBonds,
                                          implicitSolvent=app.OBC2)

        # create and add our extra forces
        # TODO

        self.integrator = mm.LangevinIntegrator(self.temperature*u.kelvin,
                                           1.0/u.picoseconds,
                                           2.0*u.femtoseconds)
        self.simulation = app.Simulation(prmtop.topology,
                                         self.system,
                                         self.integrator)
        self.simulation.context.setPositions(inpcrd.positions)

        print('Minimizing energy for replica {}'.format(self.output_index))
        self.simulation.minimizeEnergy(maxIterations=50)

        state = self.simulation.context.getState(getPositions=True,
                                                 getVelocities=True)
        self.pos = state.getPositions()
        self.vel = state.getVelocities()

        # setup reporters to write everything to disk
        self.simulation.reporters.append(
            app.PDBReporter('output_{}.pdb'.format(self.output_index),
                                                         self.nsteps))


    @property
    def params(self):
        return np.array([self.temperature, self.k])

    @params.setter
    def params(self, new_params):
        old_temp = self.temperature
        self.temperature, self.k = new_params

        self.integrator.setTemperature(self.temperature)
        scale = math.sqrt(self.temperature / old_temp)
        self.vel *= scale

    @property
    def x(self):
        return self.pos, self.vel, self.temperature

    @x.setter
    def x(self, state):
        self.pos, self.vel, t = state
        self.vel *= math.sqrt(self.temperature / t)

    def update(self):
        print('Running replica {}'.format(self.output_index))
        self.simulation.context.setPositions(self.pos)
        self.simulation.context.setVelocities(self.vel)
        self.simulation.step(self.nsteps)
        state = self.simulation.context.getState(getPositions=True,
                                                 getVelocities=True)
        self.pos = state.getPositions()
        self.vel = state.getVelocities()

    def get_energy(self):
        self.simulation.context.setPositions(self.pos)
        s = self.simulation.context.getState(getEnergy=True)
        E = s.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
        return E / (R * self.temperature)

    def get_derivs(self):
        E = self.get_energy()
        dT = -E / self.temperature
        return np.array([dT, 0.0])

    def get_trial_energy(self, state):
        pos, vel, _ = state
        self.simulation.context.setPositions(pos)
        s = self.simulation.context.getState(getEnergy=True)
        E = s.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
        return E / (R * self.temperature)

    def get_trial_derivs(self, state):
        E = self.get_trial_energy(state)
        dT = -E / self.temperature
        return np.array([dT, 0.0])
