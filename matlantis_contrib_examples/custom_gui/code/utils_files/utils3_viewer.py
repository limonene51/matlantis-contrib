# 1. module
import nglview as nv
import numpy as np
import matplotlib.pyplot as plt
import ase
from ase import Atoms
from ase.visualize import view
from pfcc_extras.visualize.addeditor import AddEditor

# 2. function
def view_func(atoms):
    v = view(atoms, viewer='ngl')
    v.view.add_representation("ball+stick")
    display(v)


def AddEditor_func(atoms):
    v=AddEditor(atoms)
    return v


def add_lotsbonds(mols : ase.Atoms, view :  nv.widget.NGLWidget , rap_n = 200,  nglmax_n = 400 , lineOnly = False ):
    """Used in litsatoms_view function
    """
    mols = mols[np.argsort(mols.positions[:,2])]
    myslices =  [mols[i : i+nglmax_n] for i in range(0, len(mols), nglmax_n-rap_n) ] 
    for myslice in myslices:
        structure = nv.ASEStructure(myslice)
        st = view.add_structure(structure,defaultRepresentation="")
        if lineOnly:
            st.add_ball_and_stick(lineOnly = True)
        else:
            st.add_ball_and_stick(cylinderOnly = True) #, linewidth=4


def lotsatoms_view(mols,radiusScale=0.3, rap_n = 200,  nglmax_n = 400 , lineOnly = False):
    """NGLview defaults to displaying only around 400 bonds. This is addressed by splitting and drawing.
    Sort by z-axis and draw bonds while overlapping.
    """    
    view = nv.NGLWidget(width=str(500)+ "px" ,height=str(500)+"px")
    structure = nv.ASEStructure(mols)
    #まずはspacefillのみ描く　
    view.add_structure(structure,defaultRepresentation="")
    view.add_spacefill(radiusType='covalent',radiusScale=radiusScale)
    view.center(component=0)
    view.add_unitcell()
    view.background ="#222222"
    view.camera  ="orthographic"
    add_lotsbonds(mols, view , rap_n = rap_n  , nglmax_n = nglmax_n , lineOnly = lineOnly )    
    return view


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
view_func.is_type = "viewer"
lotsatoms_view.is_type = "viewer"
get_energy_plot.is_type = "viewer"
AddEditor_func.is_type = "viewer"

# 3.3 Specified the function here that is not displayed in CustomGUI class.
add_lotsbonds.is_type = None

# 3.4 Specify the argument here that you want to select by dropdown.
# print_test.dropdown_argument = ["dropdown_arg"]

# 3.5 Specify the argument here that you want to set all_atoms as default.
# print_test.all_atoms_argument = ["atoms_all_arg"]