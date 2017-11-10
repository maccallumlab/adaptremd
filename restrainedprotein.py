from __future__ import print_function
from simtk import openmm as mm
from simtk.openmm import app
import simtk.unit as u
import numpy as np
import collections
import math
from meld.system.openmm_runner import cmap


# ideal gas constant
R = 8.314e-3


tors_template = (
    '0.5 * k_tors * eff^2;'
    'eff = step(-diff - delta) * (diff + delta) + step(diff - delta) * (diff - delta);'
    'diff = 3.14159 - abs(abs(theta - theta0) - 3.14159);')

dist_template = (
    'E1 + E2 + E3 + E4;'
    'E1 = step(r1-r) * (0.5*{k}*(r1-r2)^2 + {k}*(r2-r1)*(r1-r));'
    'E2 = step(r-r1)*step(r2-r) * 0.5*{k}*(r-r2)^2;'
    'E3 = step(r-r3)*step(r4-r) * 0.5*{k}*(r-r3)^2;'
    'E4 = step(r-r4) * (0.5*{k}*(r4-r3)^2 + {k}*(r4-r3)*(r-r4));'
)

class RestrainedProtein(object):
    def __init__(self, output_index, parm_path, crd_path, init_temp, init_log_k,
                 variable_springs, fixed_springs, fixed_torsions, nsteps=500,
                 output_steps=500, gpuid=None):
        self.output_index = output_index
        self.nsteps = nsteps
        self.temperature = init_temp
        self.log_k = init_log_k
        self.k = np.exp(self.log_k)

        prmtop = app.AmberPrmtopFile(parm_path)
        inpcrd = app.AmberInpcrdFile(crd_path)

        self.system = prmtop.createSystem(nonbondedMethod=app.CutoffNonPeriodic,
                                          nonbondedCutoff=1.8*u.nanometer,
                                          constraints=app.HBonds,
                                          implicitSolvent=app.OBC2)

        # Add the AMAP correction
        with open(parm_path) as topfile:
            mapper = cmap.CMAPAdder(topfile.read())
        mapper.add_to_openmm(self.system)

        #
        # create and add our extra forces
        #

        # add torsions
        torsion_force = mm.CustomTorsionForce(tors_template)
        torsion_force.addGlobalParameter('k_tors', 20.0)
        torsion_force.addPerTorsionParameter('theta0')
        torsion_force.addPerTorsionParameter('delta')
        for i, j, k, l, theta0, delta in fixed_torsions:
                torsion_force.addTorsion(i, j, k, l, [theta0, delta])
        self.system.addForce(torsion_force)

        # add fixed bonds
        fixed_bond_force = mm.CustomBondForce(dist_template.format(k='k_fixed'))
        fixed_bond_force.addGlobalParameter('k_fixed', 250.0)
        fixed_bond_force.addPerBondParameter('r1')
        fixed_bond_force.addPerBondParameter('r2')
        fixed_bond_force.addPerBondParameter('r3')
        fixed_bond_force.addPerBondParameter('r4')
        for i, j, r1, r2, r3, r4 in fixed_springs:
            fixed_bond_force.addBond(i, j, [r1, r2, r3, r4])
        self.system.addForce(fixed_bond_force)

        # add variable bonds
        for i, bond_list in enumerate(variable_springs):
            kstring = 'k_bond{}'.format(i)
            bf = mm.CustomBondForce(dist_template.format(k=kstring))
            bf.addGlobalParameter(kstring, self.k[i])
            bf.addPerBondParameter('r1')
            bf.addPerBondParameter('r2')
            bf.addPerBondParameter('r3')
            bf.addPerBondParameter('r4')

            for i, j, r1, r2, r3, r4 in bond_list:
                bf.addBond(i, j, [r1, r2, r3, r4])
            self.system.addForce(bf)

        self.integrator = mm.LangevinIntegrator(self.temperature*u.kelvin,
                                           1.0/u.picoseconds,
                                           2.0*u.femtoseconds)

        platform = mm.Platform.getPlatformByName('CUDA')
        properties = {'CudaDeviceIndex': str(gpuid),
                      'CudaPrecision': 'mixed'}
        self.simulation = app.Simulation(prmtop.topology,
                                         self.system,
                                         self.integrator,
                                         platform,
                                         properties)

        self.simulation.context.setPositions(inpcrd.positions)

        print('Minimizing energy for replica {}'.format(self.output_index))
        self.simulation.minimizeEnergy()

        state = self.simulation.context.getState(getPositions=True,
                                                 getVelocities=True)
        self.pos = state.getPositions(asNumpy=True)
        self.vel = state.getVelocities(asNumpy=True)

        # setup reporters to write everything to disk
        self.simulation.reporters.append(
            app.PDBReporter('output_{}.pdb'.format(self.output_index), output_steps))


    @property
    def params(self):
        return np.array([self.temperature] + list(self.log_k))

    @params.setter
    def params(self, new_params):
        old_temp = self.temperature
        self.temperature = new_params[0]
        self.log_k = new_params[1:]
        self.k = np.exp(self.log_k)
        self._update_params(old_temp)

    def _update_params(self, old_temp):
        self.integrator.setTemperature(self.temperature)
        for i, val in enumerate(self.k):
            self.simulation.context.setParameter('k_bond{}'.format(i), val)

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
        self.simulation.context.setPositions(self.pos)
        self.simulation.context.setVelocities(self.vel)
        self.simulation.step(self.nsteps)
        state = self.simulation.context.getState(getPositions=True,
                                                 getVelocities=True)
        self.pos = state.getPositions(asNumpy=True)
        self.vel = state.getVelocities(asNumpy=True)

    def get_energy(self):
        return self.get_trial_energy((self.pos, self.vel, self.temperature))

    def get_derivs(self):
        return self.get_trial_derivs((self.pos, self.vel, self.temperature))

    def get_trial_energy(self, state):
        pos, vel, _ = state
        self.simulation.context.setPositions(pos)
        s = self.simulation.context.getState(getEnergy=True)
        E = s.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
        return E / (R * self.temperature)

    def get_trial_derivs(self, state):
        # set the positions
        pos, vel, _ = state
        self.simulation.context.setPositions(pos)

        # derivative wrt temperature
        s = self.simulation.context.getState(getEnergy=True)
        E = s.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
        dT = -E / (R * self.temperature**2)

        #
        # derivative wrt force constants
        #

        # get the energy without force constants
        old_k = self.k.copy()
        self.k = np.zeros_like(self.k)
        self._update_params(self.temperature)
        s = self.simulation.context.getState(getEnergy=True)
        Eref = s.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)

        # get the energy for each force constant in turn
        derivs = np.zeros_like(old_k)
        for i in range(len(old_k)):
            p = np.zeros_like(old_k)
            p[i] = 1.0
            self.k = p
            self._update_params(self.temperature)
            s = self.simulation.context.getState(getEnergy=True)
            Ep = s.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
            derivs[i] = old_k[i] * (Ep - Eref) / (R * self.temperature)

        self.k = old_k
        self._update_params(self.temperature)
        return np.array([dT] + list(derivs))
