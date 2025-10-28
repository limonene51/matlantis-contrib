# 1. module
import numpy as np
import functools
import matplotlib.pyplot as plt
import nglview as nv

import ase
from ase import Atoms
from ase.build import surface, add_adsorbate
from ase.constraints import FixAtoms, ExpCellFilter
from ase.optimize import BFGS

from matlantis_features.ase_ext.optimize import FIRELBFGS
from pfcc_extras.visualize.addeditor import AddEditor
from pfcc_extras.liquidgenerator.liquid_generator import LiquidGenerator
from pfcc_extras.structure.ase_rdkit_converter import smiles_to_atoms, generate_conformers

# 2. function
def opt_cell_size(m,sn = 10, iter_count = False):
    m.set_constraint() # clear constraint
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


def makesurface(
    atoms: Atoms, miller_indices=(1, 1, 1), layers=4, rep=[4, 4, 1]
) -> Atoms:
    s1 = surface(atoms, miller_indices, layers)
    s1.center(vacuum=10.0, axis=2)
    s1 = s1.repeat(rep)
    s1.set_positions(
        s1.get_positions() - [0, 0, min(s1.get_positions()[:, 2])]
    )
    s1.pbc = True
    return s1


def f_set_constraints(atoms, indices):
    c = FixAtoms(indices=indices)
    atoms.set_constraint(c)
    print(f'Set constraints for indices = {indices}')
    return atoms


def f_add_adsorbate(slab,adsorbate, height, position=(0, 0), offset=None, mol_index=0):
    add_adsorbate(slab,adsorbate, height, position=position, offset=offset, mol_index=mol_index)
    return slab


def f_liquid_generator(density=1.0,
                       composition=[
                           {"name": "water", "structure": "O", "number": 20, "density": 1, "tag": 1},
                           {"name": "ethanol", "smiles": "CCO", "number": 20, "density": 0.789}, 
                       ],
                      packmol_bin="./packmol/packmol",
                      output_name="mixture_001.xyz",
                      output_dir="intermediate",
                      calc_pbc=True):
    params = {
        "density":density,
        "composition": composition,
        "packmol_bin": packmol_bin,
        "output_name": output_name,
        "output_dir": output_dir,
        "calc_pbc": calc_pbc
    }

    generator = LiquidGenerator(**params)
    atoms = generator.run()
    return atoms


def myopt(atoms, fmax = 0.001):
    print(atoms.calc.estimator.calc_mode, "/", atoms.calc.estimator.model_version)
    opt=FIRELBFGS(atoms)
    opt.run(fmax=fmax)
    return atoms


def AddEditor_func(atoms):
    v=AddEditor(atoms)
    return v


def get_energy_plot(atoms):
    ene = {}
    for i, tmp_atoms in enumerate(atoms):
        # tmp_atoms.calc = calculator
        ene[i] = tmp_atoms.get_potential_energy()

    fig, ax = plt.subplots()
    ax.plot(list(ene.keys()), list(ene.values()))
    ax.set_xlabel("Frame")
    ax.set_ylabel("Potential Energy")
    ax.set_title("Potential Energy over Frames")

    return fig 

# 3. Advanced settings
# 3.1 (Planned Feature Additions)

# 3.2 Specified the function here that is viewer.
get_energy_plot.is_type = "viewer"
AddEditor_func.is_type = "viewer"

# 3.3 Specified the function here that is not displayed in CustomGUI class.
# add_lotsbonds.is_type = None

# 3.4 Specify the argument here that you want to select by dropdown.
f_add_adsorbate.dropdown_argument = ["adsorbate"]

# 3.5 Specify the argument here that you want to set all_atoms as default.
# print_test.all_atoms_argument = ["atoms_all_arg"]