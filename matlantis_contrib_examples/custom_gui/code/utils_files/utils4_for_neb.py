# 1. module
import numpy as np
import math
import matplotlib.pyplot as plt

from ase.build import surface, add_adsorbate
from ase.cluster.cubic import FaceCenteredCubic
from ase.constraints import FixAtoms, ExpCellFilter
from ase.mep.neb import NEB
from ase.optimize import BFGS, FIRE
from ase.build import sort

from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
from pfp_api_client.pfp.estimator import Estimator
from matlantis_features.ase_ext.optimize import FIRELBFGS

from pfcc_extras.visualize.addeditor import AddEditor
from pfcc_extras.structure.ase_rdkit_converter import smiles_to_atoms
from pfcc_extras.structure.connectivity import get_connectivity_matrix

# 2. function
def f_neighbor_atom(atoms, index):
    """ Get neighbor_atom for np.array

    Returns:
        np.array
    """
    return np.where(get_connectivity_matrix(atoms).toarray()[index] !=0)[0]


def f_rotate_open_space_to_minus_z(atoms, index):
    """
    Atoms rotated so that, for the specified index,
    the direction with the most spatial clearance is parallel to the z-axis.
    
    Returns:
        atoms: Atoms rotated to be parallel to the z-axis
    """
    new_atoms = atoms.copy()
    atoms_pos = atoms.get_positions()
    idx_neibor_atoms = f_neighbor_atom(atoms, index)
    
    # Get sum vector for all neighbor atoms
    vec_sum=0
    for i in range(len(idx_neibor_atoms)):
        vec_sum += atoms_pos[idx_neibor_atoms[i]] - atoms_pos[index]
    
    vec_z = [0,0,1]
    
    theta = np.arccos(np.inner(vec_sum, vec_z) /
                      np.linalg.norm(vec_sum) /
                      np.linalg.norm(vec_z)) * 180 / math.pi
    
    new_atoms.rotate(a=theta, v=np.cross(vec_sum, vec_z), center=atoms_pos[index])

    return new_atoms


def f0_smiles_to_atoms(smiles="[C-]#[O+]", index_orient_z=0):
    """
    Args:
        smiles: smiles
        index_orient_z (int or False): rotate to be parallel to the z-axis for specified index. If False, Atoms isn't rotated.

    Returns:
        Atoms
    """
    atoms = smiles_to_atoms(smiles)
    if type(index_orient_z) == int:
        atoms = f_rotate_open_space_to_minus_z(atoms, index_orient_z)

    print("Please write atoms as 'mol.json'.")
    return atoms


def f1_opt_cell_size(m,sn = 10, iter_count = False):
    """opt_cell_size
    """
    m.set_constraint() # clear constraint
    # m.set_calculator(calculator)
    maxf = np.sqrt(((m.get_forces())**2).sum(axis=1).max())
    ucf = ExpCellFilter(m)
    print("ini   pot:{:.4f},maxforce:{:.4f}".format(m.get_potential_energy(),maxf))
    de = -1 
    s = 1
    ita = 50
    while ( de  < -0.01 or de > 0.01 ) and s <= sn :
        opt = BFGS(ucf,maxstep=0.04*(0.9**s),logfile=None)
        old  =  m.get_potential_energy() 
        opt.run(fmax=0.005,steps =ita)
        maxf = np.sqrt(((m.get_forces())**2).sum(axis=1).max())
        de =  m.get_potential_energy()  - old
        print("{} pot:{:.4f},maxforce:{:.4f},delta:{:.4f}".format(s*ita,m.get_potential_energy(),maxf,de))
        s += 1
    if iter_count == True:
        return m, s*ita
    else:
        return m


def f1_1_repeat_bulk(bulk, repeat=[2, 2, 2]):
    """repeat
    """
    bulk = bulk.repeat(repeat)
    bulk = sort(bulk)
    # Shift positions a bit to prevent surface to be cut at wrong place when `makesurface` is called
    bulk.positions += [0.01, 0, 0]

    return bulk


def f2_makesurface(atoms,miller_indices=(1,1,1),layers=2,rep=[1,1,1]):
    """make surface
    """
    s1 = surface(atoms, miller_indices,layers)
    s1.center(vacuum=10.0, axis=2)
    s1 = s1.repeat(rep)
    s1.set_positions(s1.get_positions() - [0,0,min(s1.get_positions()[:,2])])
    s1.pbc = True
    return s1


def f3_set_constraints(atoms, height=3.0):
    indices = [atom.index for atom in atoms if atom.position[2] <= height]
    c = FixAtoms(indices=indices)
    atoms.set_constraint(c)
    print(f'Set constraints for indices = {indices}')
    return atoms


def f4_add_adsorbate(slab,adsorbate, height=2.3, position=(8.16, 4.70), offset=None, mol_index=0):
    add_adsorbate(slab,adsorbate, height, position=position, offset=offset, mol_index=mol_index)
    return slab


def f5_myopt(atoms, fmax = 0.001):
    print(atoms.calc.estimator.calc_mode, "/", atoms.calc.estimator.model_version)
    opt=FIRELBFGS(atoms)
    opt.run(fmax=fmax)
    print("Please write atoms as 'FS.json'.")
    return atoms


def f6_AddEditor(atoms):
    """In this example, pressing Z+ multiple times allows us to separate
       the adsorbed molecule from the slab and create the initial structure.
    """

    v=AddEditor(atoms)
    return v


def f7_myopt(atoms, fmax = 0.001):
    print(atoms.calc.estimator.calc_mode, "/", atoms.calc.estimator.model_version)
    opt=FIRELBFGS(atoms)
    opt.run(fmax=fmax)
    print("Please write atoms as 'IS.json'.")
    return atoms


def f8_neb_preparation(IS,FS,calculator, height=3.0, opt_fmax=0.005):
    """To perform NEB calculations, the initial and final structures for NEB
       are combined into a single atoms object
    """

    c = FixAtoms(indices=[atom.index for atom in IS if atom.position[2] <= height])
    IS.calc = calculator
    IS.set_constraint(c)
    BFGS_opt = BFGS(IS)
    BFGS_opt.run(fmax=opt_fmax)
    print(f"IS {IS.get_potential_energy()} eV")
    
    c = FixAtoms(indices=[atom.index for atom in FS if atom.position[2] <= height])
    FS.calc = calculator
    FS.set_constraint(c)
    BFGS_opt = BFGS(FS)
    BFGS_opt.run(fmax=opt_fmax)
    print(f"FS {FS.get_potential_energy()} eV")

    return [IS,FS]

def f9_neb_calc(atoms, calculator, beads=21, steps=2000, fmax=0.1):
    IS=atoms[0]
    FS=atoms[1]
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

def f10_show_neb_result(configs):
    energies = [config.get_total_energy() for config in configs]
    
    plt.plot(range(len(energies)),energies)
    plt.xlabel("replica")
    plt.ylabel("energy [eV]")
    plt.xticks(np.arange(0, len(energies), 2))
    plt.grid(True)
    plt.show()

# 3. Advanced settings
# 3.1 (Planned Feature Additions)

# 3.2 Specify the function here that is viewer.
f6_AddEditor.is_type = "viewer"
f10_show_neb_result.is_type = "viewer"

# 3.3 Specified the function here that is not displayed in CustomGUI class.
f_neighbor_atom.is_type = None
f_rotate_open_space_to_minus_z.is_type = None

# 3.4 Specify the argument here that you want to select by dropdown.
f4_add_adsorbate.dropdown_argument = ["adsorbate"]
f8_neb_preparation.dropdown_argument = ["IS", "FS"]

# 3.5 Specify the argument here that you want to set all_atoms as default.
f9_neb_calc.all_atoms_argument = ["atoms"]