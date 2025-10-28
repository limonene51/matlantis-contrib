# 1. module
import matplotlib.pyplot as plt
from pfcc_extras.visualize.addeditor import AddEditor
from ase import Atoms
from ase.mep.neb import NEB
from ase.optimize import BFGS, FIRE
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
from pfp_api_client.pfp.estimator import Estimator

import os
from asap3 import EMT

from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution,Stationary
from ase.md.verlet import VelocityVerlet
from ase.md import MDLogger
from ase import units
from time import perf_counter
import numpy as np
from ase.io import write, Trajectory

# 2. function
def neb_calc(IS, FS, calculator, beads=21, steps=2000, fmax=0.05):
    IS.calc = calculator
    FS.calc = calculator

    b0 = IS.copy()
    b1 = FS.copy()
    configs = [b0.copy() for i in range(beads-1)] + [b1.copy()]
    for config in configs:
        # Calculator must be set separately with NEB parallel=True, allowed_shared_calculator=False.
        # estimator = Estimator(calc_mode=EstimatorCalcMode.CRYSTAL_PLUS_D3, model_version="v8.0.0")
        estimator  = Estimator(calc_mode=IS.calc.estimator.calc_mode,
                               model_version=IS.calc.estimator.model_version)  
        calculator = ASECalculator(estimator)
        config.calc = calculator

    neb = NEB(configs, k=0.05, parallel=True, climb=True, allow_shared_calculator=False)   
    neb.interpolate()

    relax = FIRE(neb)
    
    # fmax<0.05 recommended. It takes time when it is smaller.
    # 1st NEB calculation can be executed with loose condition (Ex. fmax=0.2),
    # and check whether reaction path is reasonable or not.
    # If it is reasonable, run 2nd NEB with tight fmax condition.
    # If the reaction path is abnormal, check IS, FS structure.
    relax.run(fmax=fmax, steps=steps)

    return configs

def nvt_ast6_1(bulk_conditions={"name":"Al", "crystalstructure":"fcc", "a":4.3, "cubic":True},
               time_step = 1.0,
               temperature = 1600,     # Temperature in Kelvin
               num_md_steps = 100000,   # Total number of MD steps
               num_interval = 1000,
               output_filename = "./output/ch6/liquid-Al_NVE_1.0fs_test"
              ):
    """Atomistic Simulation Tutorial 6.1 NVT
    """
    calculator = EMT()
    # Set up a fcc-Al crystal
    atoms = bulk(**bulk_conditions)
    atoms.pbc = True
    atoms *= 3
    print("atoms = ",atoms)
    
    # Set calculator (EMT in this case)
    atoms.calc = calculator
    
    # input parameters
    # time_step    = 1.0      # MD step size in fsec
    # temperature  = 1600     # Temperature in Kelvin
    # num_md_steps = 100000   # Total number of MD steps
    # num_interval = 1000     # Print out interval for .log and .traj
    
    # Set the momenta corresponding to the given "temperature"
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature,force_temp=True)
    Stationary(atoms)  # Set zero total momentum to avoid drifting
    
    # Set output filenames
    # output_filename = "./output/ch6/liquid-Al_NVE_1.0fs_test"
    if len(output_filename.split("/")) != 1:
        mk_foldername = ("/").join(output_filename.split("/")[:-1])
        os.makedirs(mk_foldername, exist_ok=True)
    ########################################################################
    
    log_filename = output_filename + ".log"
    print("log_filename = ",log_filename)
    traj_filename = output_filename + ".traj"
    print("traj_filename = ",traj_filename)
    
    # Remove old files if they exist
    if os.path.exists(log_filename): os.remove(log_filename)
    if os.path.exists(traj_filename): os.remove(traj_filename)
    
    # Define the MD dynamics class object
    dyn = VelocityVerlet(atoms, 
                         time_step * units.fs,
                         trajectory = traj_filename,
                         loginterval=num_interval
                        )  
    
    # Print statements
    def print_dyn():
        imd = dyn.get_number_of_steps()
        time_md = time_step*imd
        etot  = atoms.get_total_energy()
        ekin  = atoms.get_kinetic_energy()
        epot  = atoms.get_potential_energy()
        temp_K = atoms.get_temperature()
        print(f"   {imd: >3}     {etot:.9f}     {ekin:.9f}    {epot:.9f}   {temp_K:.2f}")
    
    dyn.attach(print_dyn, interval=num_interval)
    
    # Set MD logger
    dyn.attach(MDLogger(dyn, atoms, log_filename, header=True, stress=False,peratom=False, mode="w"), interval=num_interval)
    
    # Now run MD simulation
    print(f"\n    imd     Etot(eV)    Ekin(eV)    Epot(eV)    T(K)")
    dyn.run(num_md_steps)
    
    print("\nNormal termination of the MD run!")

    return Trajectory(traj_filename)


def show_result(configs):
    energies = [config.get_total_energy() for config in configs]
    
    plt.plot(range(len(energies)),energies)
    plt.xlabel("replica")
    plt.ylabel("energy [eV]")
    plt.xticks(np.arange(0, len(energies), 2))
    plt.grid(True)
    plt.show()

# 3. Advanced settings (for more details, refer to the notebook)
# 3.1 (Planned Feature Additions)

# 3.2 Specify the function here that is viewer.
show_result.is_type = "viewer"

# 3.3 Specified the function here that is not displayed in CustomGUI class.
# not_visible.is_type = None

# 3.4 Specify the argument here that you want to select by dropdown.
neb_calc.dropdown_argument = ["IS", "FS"]

# 3.5 Specify the argument here that you want to set all_atoms as default.
# print_test.all_atoms_argument = ["atoms_all_arg"]