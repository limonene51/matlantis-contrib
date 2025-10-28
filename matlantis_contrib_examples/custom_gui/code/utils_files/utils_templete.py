# 1. module
from pfcc_extras.visualize.addeditor import AddEditor
from ase.build import surface
from ase import Atoms

# 2. function
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


def print_test(_any, dropdown_arg, atoms_all_arg, calculator):
    print("_any: ", _any)
    print("dropdown: ", dropdown_arg)
    print("atoms_all: ", atoms_all_arg)
    print("calculator: ", calculator)


def not_visible(_any):
    """This function is not displayed in dropdown
       because 'not_visible.is_type = None' is specified at line 29.
    """
    print("not visible")


def AddEditor_func(atoms):
    """viewer
    """

    v=AddEditor(atoms)
    return v

# 3. Advanced settings (for more details, refer to the notebook)
# 3.1 (Planned Feature Additions)

# 3.2 Specify the function here that is viewer.
AddEditor_func.is_type = "viewer"

# 3.3 Specified the function here that is not displayed in CustomGUI class.
not_visible.is_type = None

# 3.4 Specify the argument here that you want to select by dropdown.
print_test.dropdown_argument = ["dropdown_arg"]

# 3.5 Specify the argument here that you want to set all_atoms as default.
print_test.all_atoms_argument = ["atoms_all_arg"]