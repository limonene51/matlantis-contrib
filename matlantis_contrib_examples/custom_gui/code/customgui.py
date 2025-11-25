import types

# import
import sys
import traceback
import numpy as np
import math
import warnings
from math import pi, sqrt
import os
import ast
import json5
import nbformat
from datetime import datetime
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode

# ase
import ase
from ase import Atom, Atoms
from ase.optimize import FIRE, LBFGS
from ase.constraints import FixAtoms
from ase.io import read, write
from ase import neighborlist
from ase.data import covalent_radii

# matlantis_features
from matlantis_features.ase_ext.optimize import FIRELBFGS

import nglview as nv
from typing import Dict, List, Optional
from ase.visualize.ngl import NGLDisplay
from IPython.display import display
from ipywidgets import (Accordion, Button, Checkbox, FloatSlider,
                        FloatRangeSlider, GridspecLayout, HBox, VBox, Dropdown,
                        Label, Output, Text, Textarea, Select,
                        BoundedIntText, BoundedFloatText, HTML, Tab,
                        RadioButtons, Image, widgets, Layout)
from traitlets import Bunch
from nglview.widget import NGLWidget
from pfcc_extras.visualize.ngl_utils import (add_force_shape, add_axes_shape,
                                             get_struct, save_image,
                                             update_tooltip_atoms)
from pfcc_extras.structure.ase_rdkit_converter import smiles_to_atoms

import inspect
import IPython
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
import importlib
from pfcc_extras.visualize import view as pfcc_view

class CustomGUI:
    """ GUI that can excute arbitary functions.
    Author: Toshiya Sato, Matlantis Corp.(JP)
    
    """

    struct: List[Dict]  # structure used for nglview drawing.
    pot: float  # potential energy
    mforce: float  # mean average force
    show_axes: bool = False
    show_force: bool = False
    show_index: bool = False

    def __init__(self, atoms=None,
                 utils_path = "utils.py",
                 background_color="",
                 fallback_calc_mode="PBE",
                 fallback_model_version="latest",
                 bg_write_extension = "json",
                 added_import_functions = [],
                 output_height = 175,
                 output_width = 940,  # 関数の下にoutput_areaを置く場合は490
                 view_constraints_color="royalblue",
                 # w_setcalc_accordion: int = 500,
                 # w_function_accordion: int = 800,
                 h_upper_accordion = 105,  # 75だとぎりぎりのサイズ
                 w_left: int = 335,
                 w: int = 450, h: int = 470):
        """background_color: Color of nglview. Default is '#222222'(=black).
                             For example, 'White' can be used.
           fallback_calc_mode: Calc mode to be set if Atoms has no calculator.
                               Default is 'CRYSTAL_U0'.
           fallback_model_version: Model version to be set
                                    if Atoms has no calculator.
                                   Default is 'latest'.
        """

        # If no arguments are specified, output CH4.
        if atoms is None:
            atoms = smiles_to_atoms("C")

        elif type(atoms) is str:
            atoms = smiles_to_atoms(atoms)

        ##############################
        if type(atoms)==list:
            self.all_atoms = atoms
        else:
            self.all_atoms = [atoms]
        ##############################

        """Obtain the original calculator.
           If the calculator is not set, set the default calculator.
        """
        try:
            self.calc_mode = self.all_atoms[0].calc.estimator.calc_mode.value.upper()
            self.model_version = self.all_atoms[0].calc.estimator.model_version
        except AttributeError:
            self.calc_mode = fallback_calc_mode
            self.model_version = fallback_model_version
            # warnings.warn(
            #     "Calculator"
            #     f"(calc_mode={fallback_calc_mode},"
            #     f" model_version={fallback_model_version})"
            #     " was set. So, please check if there are any problems. "
            #     )

        # self.estimator = Estimator(calc_mode=self.calc_mode,
        #                            model_version=self.model_version)
        # self.atoms.calc = ASECalculator(self.estimator)
        ###############################
        
        # if type(self.all_atoms)==list:   # all_atomsは必ずリストにするようにしたからいらない
        for i in self.all_atoms:
            self.estimator = Estimator(calc_mode=self.calc_mode,
                                       model_version=self.model_version)
            i.calc=ASECalculator(self.estimator)
            
        self.atoms = self.all_atoms[0]
        # else:
        #     self.estimator = Estimator(calc_mode=self.calc_mode,
        #                                model_version=self.model_version)
        #     self.all_atoms.calc=ASECalculator(self.estimator)
        #     self.atoms = self.all_atoms
        ###############################
        self.prev_atoms = []
        self.h_upper_accordion = h_upper_accordion
        self.w_left = w_left
        self.w = w
        self.h = h
        self.vh = NGLDisplay(self.all_atoms, self.w, self.h).gui
        self.v: NGLWidget = self.vh.view
        
        # background color: distinguished by Jupyter Theme
        if background_color!="":
            self.background_color = background_color
        else:
            try:
                settings_path = os.path.expanduser(
                    "~/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings"
                )
                with open(settings_path, "r") as f:
                    settings = json5.load(f)
                
                current_theme = settings.get("theme", "Unknown")
                if current_theme=='JupyterLab Light':
                    self.background_color = "transparent"
                else:
                    self.background_color = "#222222"
    
            except Exception:
                self.background_color = "#222222"

        self.view_constraints_color = view_constraints_color

        # setting of function in utils.py
        self.output_height = output_height
        self.output_width = output_width
        self.output_area = widgets.Output(
            # layout={'border': '1px solid black', 'height': f'{self.output_height}px', 'overflow_y': 'auto', 'width':f'{self.output_width}px'}
            layout=Layout(border='1px solid black', height=f'{self.output_height}px', overflow='auto', width=f'{self.output_width}px', flex="0 0 auto")
        )
        self.function_panels = []
        self.function_names = []
        self.result_func = None
        self.bg_write_extension = bg_write_extension
        self.added_import_functions = added_import_functions

        # 作業用ipynbとテスト用pyファイルで変更するところ #################
        self.utils_path = utils_path  # ipynb
        # self.utils_path = (os.path.dirname(__file__)
        #                    + "/" + utils_path)  # py
        ###############################################################

        self.viewer_functions_list = []

        self.undo_list = []  # List to retrieve atoms from previous operation
        self.redo_list = []  # List to retrieve atoms from next operation
        self.undo_frame_list = {i:[] for i in range(self.v.max_frame+1)}
        self.redo_frame_list = {i:[] for i in range(self.v.max_frame+1)}
        self.frame_num = 0

        update_tooltip_atoms(self.v, self.atoms)
        self.recont()  # Add controller
        self.set_representation()
        self.set_atoms()
        self.pots = []
        self.traj = []

        self.cal_nnp()

        # --- force related attributes ---
        self.show_force = False
        self._force_components = []
        self.force_scale = 0.4
        self.force_color = [1, 0, 0]

        self.show_axes = False
        self._axes_components = []
        self.axes_scale = 0.4
        self.axes_color = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def clear_force(self):
        # Remove existing force components.
        for c in self._force_components:
            self.v.remove_component(c)  # Same with c.clear()
        self._force_components = []

    def clear_axes(self):
        # Remove existing axes components.
        for c in self._axes_components:
            self.v.remove_component(c)  # Same with c.clear()
        self._axes_components = []

    def add_force(self):
        try:
            c = add_force_shape(self.atoms, self.v,
                                self.force_scale, self.force_color)
            self._force_components.append(c)
        except Exception:
            with self.vh.out_widget:
                print(traceback.format_exc(), file=sys.stderr)
            # `append_stderr` method shows same text twice somehow...
            # self.gui.out_widget.append_stderr(str(e))
            return

    def add_axes(self):
        try:
            c = add_axes_shape(self.atoms, self.v, self.axes_scale)
            self._axes_components.append(c)
        except Exception:
            with self.vh.out_widget:
                print(traceback.format_exc(), file=sys.stderr)
            # `append_stderr` method shows same text twice somehow...
            # self.gui.out_widget.append_stderr(str(e))
            return

    def recont(self):
        grid0 = GridspecLayout(2, 4, layout={"width": "610px"})
        self.vh.xplusview = Button(description="View X+")
        self.vh.yplusview = Button(description="View Y+")
        self.vh.zplusview = Button(description="View Z+")
        self.vh.xminusview = Button(description="View X-")
        self.vh.yminusview = Button(description="View Y-")
        self.vh.zminusview = Button(description="View Z-")
        self.vh.xplusview.on_click(self.rotate_view)
        self.vh.yplusview.on_click(self.rotate_view)
        self.vh.zplusview.on_click(self.rotate_view)
        self.vh.xminusview.on_click(self.rotate_view)
        self.vh.yminusview.on_click(self.rotate_view)
        self.vh.zminusview.on_click(self.rotate_view)
        grid0[0, 0] = self.vh.xplusview
        grid0[0, 1] = self.vh.yplusview
        grid0[0, 2] = self.vh.zplusview
        grid0[1, 0] = self.vh.xminusview
        grid0[1, 1] = self.vh.yminusview
        grid0[1, 2] = self.vh.zminusview

        self.vh.camera_reset = Button(description="View Reset")
        # self.vh.camera_reset.on_click(
        #     lambda b: self.v.control.orient(self.v.center())
        # )      
        self.vh.camera_reset.on_click(self.reset_view)      
        
        grid0[0, 3] = self.vh.camera_reset

        self.vh.selected_atoms_label = Label("Selected atoms:")
        self.vh.selected_atoms_textarea = Textarea(layout={"width": "295px"})
        self.selected_atoms_hbox = HBox(
            [self.vh.selected_atoms_label, self.vh.selected_atoms_textarea]
        )

        """Setatoms per x,y,z axis
        """
        self.vh.setatoms_x = FloatRangeSlider(min=-50, max=50, step=0.1,
                                              value=[-50, 50],
                                              layout={"width": "375px"},
                                              description="atoms x:")

        self.vh.setatoms_x.observe(self.set_atoms)
        self.vh.setatoms_y = FloatRangeSlider(min=-50, max=50, step=0.1,
                                              value=[-50, 50],
                                              layout={"width": "375px"},
                                              description="atoms y:")

        self.vh.setatoms_y.observe(self.set_atoms)
        self.vh.setatoms_z = FloatRangeSlider(min=-50, max=50, step=0.1,
                                              value=[0, 50],
                                              layout={"width": "375px"},
                                              description="atoms z:")

        self.vh.setatoms_z.observe(self.set_atoms)

        """Display in "selected atoms" only selected elements
        """
        self.vh.display_only_selected_elements = RadioButtons(
            options=['All', 'Only selected elements'],
            value="All", disabled=False)


        """Undo and Redo for Add_mole function
        """
        self.vh.undo = Button(icon="rotate-left",layout={"width": "70px"},
                              tooltip="Undo",
                              # description="Undo ↰"
                             )
        self.vh.undo.on_click(self.undo)
        self.vh.redo = Button(icon="rotate-right", layout={"width": "70px"},
                              tooltip="Redo",
                             # description="Redo ↱"
                             )
        self.vh.redo.on_click(self.redo)

        self.vh.undoredo_for_all_atoms = Checkbox(description="Undo/Redo for all atoms", layout={"width": "90px"})  # これ作っただけでまだなにもやってない

        """Clear Textarea
        """
        self.vh.cleartextarea = Button(description="",
                                       icon="broom",
                                       layout={"width":"58px"},
                                       tooltip="Clear Selected atoms")
        self.vh.cleartextarea.on_click(self.clear_textarea)

        """Set calculator
        """
        self.vh.setcalc_calculator = Dropdown(
            options=list(EstimatorCalcMode.__members__.keys()),
            description="calc_mode", value=self.calc_mode,
            layout={'width': '275px'}
        )

        self.vh.setcalc_version = Dropdown(
            options=Estimator().available_models,
            description="version", value=self.model_version,
            layout={'width': '175px'}
        )

        # self.grid_setcalc = GridspecLayout(1, 3, layout={"width": "500px"})
        # self.grid_setcalc[0, 1] = self.vh.setcalc_calculator
        # self.grid_setcalc[0, 2] = self.vh.setcalc_version

        self.vh.setcalc_calculator.observe(self.on_calculator_change, names="value")
        self.vh.setcalc_version.observe(self.on_calculator_change, names="value")

        """Opt attributes
        """
        self.vh.nnptext = Textarea(disabled=True,
                                   layout={"width": "210px", "height": "33px"})

        # layout is adjusted width of step and fmax

        self.vh.show_force_checkbox = Checkbox(
            value=self.show_force,
            description="Show force",
            indent=False,
            align_self='center',
            layout={'width': '220px'},
        )
        self.vh.show_force_checkbox.observe(self.show_force_event)

        self.vh.show_axes_checkbox = Checkbox(
            value=self.show_axes,
            description="Show axes",
            indent=False,
            align_self='center',
            layout={'width': '220px'},
        )
        self.vh.show_axes_checkbox.observe(self.show_axes_event)

        # check ############################################
        """Show index
        """
        self.vh.show_index_checkbox = Checkbox(
            value=self.show_index,
            description="Show index",
            indent=False,
            align_self='center',
            layout={'width': '220px'},
        )
        self.vh.show_index_checkbox.observe(self.show_index_event)

        """Show index in "Selected atoms"
        """
        self.vh.show_index_one_part_checkbox = Checkbox(
            # value=self.show_index,
            description="Show index in 'Selected atoms'",
            indent=False,
            align_self='center',
            layout={'width': '220px'},
        )
        self.vh.show_index_one_part_checkbox.observe(self.show_index_event)

        """Show constraints
        """
        self.vh.show_constraints_checkbox = Checkbox(
            # value=self.show_index,
            # value=True,
            description="Show constraints",
            indent=False,
            align_self='center',
            layout={'width': '220px'},
        )
        self.vh.show_constraints_checkbox.observe(self.show_constraints_event)

        self.grid_cb = GridspecLayout(3, 2, layout={"width":f"{self.w}px"})
        self.grid_cb[0, 0] = self.vh.show_force_checkbox
        self.grid_cb[0, 1] = self.vh.show_index_checkbox
        self.grid_cb[1, 0] = self.vh.show_axes_checkbox
        self.grid_cb[1, 1] = self.vh.show_index_one_part_checkbox
        self.grid_cb[2, 0] = self.vh.show_constraints_checkbox
        ####################################################

        self.vh.out_widget = Output(layout={"border": "0px solid black"})

        self.vh.update_display = Button(
            description="update_display",
            tooltip="Refresh display. It can be used \
                     when target atoms is updated in another cell..",
        )
        self.vh.update_display.on_click(self.update_display)


        """ Read and Write
        """
        # atoms file uploader
        browser_width = 300  # 420
        self.uploader = FileBrowser(description="", view_type="select", w=browser_width, h=380)
        self.load_button = Button(description="Read", layout={"width":"100px"})
        self.load_button.on_click(self.on_load)

        self.reload_uploader_button = Button(description="",
                                             tooltip="reload folder",
                                             icon="refresh",
                                             layout=Layout(width='30px', height='30px', padding='0px'),
                                             button_style="")
        # self.reload_uploader_button.style.button_color = "transparent"
        self.reload_uploader_button.on_click(self.on_reload_uploader)

        # write
        self.write_file_text = Text(description="", value="mol.json",
                                    layout={"width":f"{browser_width}px"})
        self.write_button = Button(description="Write", layout={"width":"100px"})
        self.write_button.on_click(self.on_write)

        bullet_points = "■"

        # add/copy/del frame
        # self.copy_frame_button = Button(description="Copy Frame",
        #                                 layout={"width":"150px"}
        #                                )
        # self.del_frame_button = Button(description="Del Frame",
        #                                layout={"width":"150px"}
        #                               )
        # self.add_frame_button = Button(description="Add Frame",
        #                                layout={"width":"150px"}
        #                               )
        # self.add_frame_browser = FileBrowser(description="", w=300)
        # self.help_frame_button = Button(description=f"help",
        #                                 layout={"height":"25px","width":"100px"},
        #                                 style=dict(button_color='#007ACC')
        #                                )

        # self.copy_frame_button.on_click(self.on_copy_frame)
        # self.del_frame_button.on_click(self.on_del_frame)
        # self.add_frame_button.on_click(self.on_add_frame)

        # HOME page
        original_controlbox = [i for i in list(self.vh.control_box.children)]
        if type(self.all_atoms)==list: # listのときにchildrenに出てくるIntSliderを消す
            original_controlbox = original_controlbox[:-1]
        ######################################################################

        # v_setting = VBox(
        #     [VBox([HTML(value=f"<b><font size='4'> \
        #                 {bullet_points} Viewer Setting </font><b>")]),
        #     ]
        #     + original_controlbox
        #     + [grid0]
        # )

        # tab_children = [v_setting]

        # tab_titles = {0: 'Viewer Setting'}   # ipywidgets<=7.7.1 version

        # # Just width that perfectly fits a laptop is 465px
        # self.hometab = Tab(children=tab_children,
        #                    _titles=tab_titles,
        #                    layout={"width": "505px"})

        # self.hometab.titles = ['Viewer Setting']  # ipywidgets>=8.1.3 version

        self.homechildren = VBox([
            HTML("<div style='margin-top:10px'></div>"),
            HTML(value=f"<b><font size='4'> {bullet_points} HOME </font><b>"),
            # HTML(value=f"<b><font size='4'> {'<br>'*2} {'&nbsp;'*5} home </font><b>"),
        ])

        ##################################################################

        # viewer = VBox(
        #     [HTML(value=f"<b><font size='4'> {bullet_points} \
        #                 Read or Write atoms file </font><b>"),
        #      HBox([self.uploader.dropdown, self.load_button]),
        #      HBox([self.write_file_text, self.write_button]),
        #      HTML("<div style='margin-top:15px'></div>"),
        #      VBox([HTML(value=f"<b><font size='4'> \
        #                 {bullet_points} Viewer Setting </font><b>")]),
        #     ]
        #     + original_controlbox
        #     + [grid0]
        # )

        self.function_panels.append(self.homechildren)
        self.function_names.append("HOME")

        """add button of function in util.py
        """
        # module_name = 'utils'
        module_name = self.utils_path.split("/")[0].split(".py")[0]
        
        # utils.pyをASTで解析して関数名を定義順に取得
        with open(self.utils_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        
        # 関数定義ノードを抽出し、名前を定義順に取得 (alphabetorder)
        function_names_defined_order = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]

        self.added_import_functions.reverse()
        for func in self.added_import_functions:
            function_names_defined_order.insert(func[1],func[0])
        
        # モジュールをインポート
        spec = importlib.util.spec_from_file_location(module_name, self.utils_path)
        self.utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.utils)

        # loop through functions
        self.dropdown_func_list = []
        for name in function_names_defined_order:
            attr = getattr(self.utils, name)

            if (
                callable(attr)
                and not name.startswith("__")
                # and attr.__module__ == "self.utils"
                and (name in function_names_defined_order or name in self.included_functions)
            ):
                
                setattr(self, name, attr)  # setattr(CustomGUI, name, attr)と同じ意味

                # viewer or not
                try:
                    func_type = attr.is_type
                except Exception:
                    pass
                else:
                    if func_type == "viewer":
                        self.viewer_functions_list.append(name)
                    elif func_type is None:
                        continue  # continue target is "for name in function_names_defined_order"
                    else:
                        pass

                # deal with function.atoms_argument specified in utils.py
                try:
                    dropdown_argument_list = attr.dropdown_argument
                except Exception:
                    dropdown_argument_list = []

                # deal with function
                try:
                    all_atoms_argument_list = attr.all_atoms_argument
                except Exception:
                    all_atoms_argument_list = []

                # get informations of argument
                sig = inspect.signature(attr)
                arg_widgets = {}
                arg_ui = []
                
                # loop through arguments
                for loop, (param_name, param) in enumerate(sig.parameters.items()):

                    # # old version). 引数にatomsが入っていたらself.atomsに置き換えるバージョン (molやmなど違う場合が多いので却下)
                    # if param_name == "atoms":
                    #     continue

                    # if loop == 0:  # The first argument is atoms
                    #     continue

                    if param_name in dropdown_argument_list:
                        fb = FileBrowser(description=param_name)
                        fb_dropdown = fb.dropdown
                        reload_button = widgets.Button(
                            description='',
                            icon='refresh',
                            layout=widgets.Layout(width='30px', height='30px', padding='0px'),
                            button_style=''
                        )

                        self.dropdown_func_list.append(fb)
                        reload_button.on_click(self.on_reload_uploader)
                        widget = HBox([fb_dropdown, reload_button])

                    elif param_name in all_atoms_argument_list:
                        widget = widgets.Text(
                            value="[D] all_atoms",
                            description=param_name,
                            tooltip=param_name,
                        )

                    elif param_name.lower() == "calculator":
                        widget = widgets.Text(value="[D] calculator",
                                              description=param_name,
                                              tooltip=param_name)

                    else:
                        default = param.default if param.default is not inspect.Parameter.empty else ""  # default value of argument

                        if loop == 0 and default=="":  # The default of first argument is atoms
                            widget = widgets.Text(
                                value="[D] atoms",
                                description=param_name,
                                tooltip=param_name,
                            )

                        else:
                            widget = widgets.Textarea(
                                value=str(default),  # 一旦boolをstrに変換し、関数を実行するところでboolに戻すようにする
                                description=param_name,
                                tooltip=param_name,
                            )



                        # 引数が数値だった場合はfloattextにしていたが、不要と思ったので削除
                        # if isinstance(default, bool):
                        #     widget = widgets.Text(
                        #         value=str(default),  # 一旦boolをstrに変換し、関数を実行するところでboolに戻すようにする
                        #         description=param_name,
                        #         tooltip=param_name,
                        #     )
                        # elif isinstance(default, (int, float)):
                        #     widget = widgets.FloatText(
                        #         value=default if default != "" else 0.0,
                        #         description=param_name,
                        #         tooltip=param_name,
                        #     )
                        # else:
                        #     widget = widgets.Text(
                        #         value=str(default),
                        #         description=param_name,
                        #         tooltip=param_name,
                        #     )

                    # 引数がkwargsではない場合は引数リストに加える
                    if param.kind != inspect.Parameter.VAR_KEYWORD:
                        arg_widgets[param_name] = widget
                        arg_ui.append(widget)

                # Run
                run_button = widgets.Button(description=f"Run", tooltip=attr.__doc__, layout={"width":"490px"},
                                            style=dict(button_color='#b41818', font_weight='bold', font_size='16px', text_color='white')
                                           )
                docstring_button = widgets.Button(description=f"Source code", layout={"height":"25px","width":"100px"},
                                                  style=dict(button_color='#007ACC', text_color="white")
                                                 )

                run_button.on_click(self.excute_utils_function)
                docstring_button.on_click(self.show_docstring)

                # background job
                folder_name_bgcalc = widgets.Text(value="./background_job_results",
                                                  description="Folder name to save results"
                                                 )
                output_file_name = widgets.Text(value="gui_notebook",
                                               description="Output File Name",
                                               style={'description_width': 'auto'}
                                              )
                # output_file_name = widgets.Text(value="bg_test.ipynb",
                #                                 description="Output File Name to excute job"
                #                                )
                bgcalc_button = Button(description="Create notebook for Background job",
                                       tooltip=attr.__doc__,
                                       layout={"width":"450px"},
                                       style=dict(font_weight='bold', font_size='16px')  # button_color='#b41818', 
                                      )

                bgcalc_button.on_click(self.run_background_job)
                
                bgcalc_accordion = Accordion(
                    children=[VBox([folder_name_bgcalc,
                                    HBox([output_file_name,
                                          HTML(value=".ipynb")
                                         ]),
                                    bgcalc_button
                                   ])], layout = {"width":"490px"})
        
                bgcalc_accordion.set_title(0, 'Setting of Running in Background')
        
                # Set all sections to be closed
                bgcalc_accordion.selected_index = None

                # VBox
                panel = VBox([HTML("<div style='margin-top:10px'></div>")]
                             + [HBox([HTML(value=f"<b><font size='4'> \
                                     {bullet_points} {name} function </font><b>"),
                                      HTML(value="<br>"),
                                      docstring_button])]
                             + arg_ui
                             + [HTML("<div style='margin-top:10px'></div>")]
                             + [HBox([run_button])]
                             + [HTML("<div style='margin-top:10px'></div>")]
                             + [bgcalc_accordion]
                            )
                # left_border = widgets.Box(value="",
                #                          layout=Layout(width = "2px",
                #                                        border = "solid")
                #                         )
                # panel = HBox([left_border,panel])
                
                self.function_panels.append(panel)
                self.function_names.append(name)


        selector_options = [(("🖼️ " + name) if name in self.viewer_functions_list else ("🔧 " + name), i) for i, name in enumerate(self.function_names)]

        """Function Selector
        """
        self.selector = Dropdown(
            options=selector_options,
            description="Function",
            layout={"width": "400px"},
            value=0
        )

        self.selector.observe(self.on_selector_change, names="value")
        self.on_selector_change({'new': self.selector.value})  # display children upon startup


        """File Accordion
        """
        # read_atoms_box = VBox([
        #     HTML("<b><font size='3'>📂 Read Atoms</font></b>"),
        #     HBox([self.uploader.dropdown, self.load_button]),
        #     HTML("<div style='margin-top:5px'></div>"),
        #     HTML("<b><font size='3'>🖋️ Write Atoms</font></b>"),
        #     HBox([self.write_file_text, self.write_button]),
        # ])
        read_atoms_box = VBox([
            HBox([HTML("<b><font size='3'>📂 Read Atoms</font></b>"), 
                  self.reload_uploader_button,
                  self.load_button], layout=Layout(justify_content='space-between')),
            HBox([self.uploader.dropdown]),
            HTML("<div style='margin-top:5px'></div>"),
            HBox([HTML("<b><font size='3'>🖋️ Write Atoms</font></b>"),
                  self.write_button], layout=Layout(justify_content='space-between')),
            HBox([self.write_file_text]),
        ])

        # self.read_write_accordion = Accordion(children=[read_atoms_box])
        self.read_write_accordion = Accordion(children=[read_atoms_box], layout={"width":f"{self.w_left}px"})
        self.read_write_accordion.set_title(0, "📂 File")
        self.read_write_accordion.selected_index = 0  # 初期状態で折りたたむ場合はNone

        adjust_accordion_width(
            self.read_write_accordion,
            closed_width="100px", opened_width=f"{self.w_left}px",
        )  # accordion can change size when accordion is false

        # self.read_write_accordion.observe(self.on_read_write_accordion_change, names='selected_index')

        """Upper accordion
        """
        # pad = "\u3000\u200B" * 5  # u3000は全角スペース, u200Bはゼロ幅スペース。これをタイトルの文字列のところに入れたら疑似的に改行をつくれる

        # 1. setcalc
        w_setcalc_accordion = 500
        self.setcalc_accordion = Accordion(
            children=[HBox([self.vh.setcalc_calculator,
                            self.vh.setcalc_version])],
            layout={"width":"120px",
            # layout={"width":f"{w_setcalc_accordion}px",
                    "height":f"{self.h_upper_accordion}px", "margin":"0 0 0 0px", "padding":"1px"}
        )

        self.setcalc_accordion.set_title(0, "🧪 Calculation Conditions")
        # self.setcalc_accordion.selected_index = 0  # 起動時に開く場合 (上行のlayout.widthの変更も忘れず)

        adjust_accordion_width(
            self.setcalc_accordion,
            closed_width="120px", opened_width=f"{w_setcalc_accordion}px",
            closed_height=f"{self.h_upper_accordion}px", opened_height=None,
        )  # accordion can change size when accordion is false


        # 2. function
        w_function_accordion = 800
        self.function_accordion = Accordion(
            children=[HBox([self.selector,
                            HTML(value="<div style='white-space: nowrap;'>Drag into argument: </div>", layout={"margin":"0 0 0 20px"}),
                            HTML(drag_html, layout={"margin":"0 0 0 5px"})])],
            layout={"width":"120px",
            # layout={"width":f"{self.w_function_accordion}px",
                    "height":f"{self.h_upper_accordion}px", "margin":"0 0 0 20px", "padding":"1px"}
        )

        self.function_accordion.set_title(0, "🧮 Function")
        # self.function_accordion.selected_index = 0

        adjust_accordion_width(
            self.function_accordion,
            closed_width="120px", opened_width=f"{w_function_accordion}px",
            closed_height=f"{self.h_upper_accordion}px", opened_height=None,
        )


        # 3. viewer 1 (original_controlbox)
        w_v1 = 500
        self.viewer1_accordion = Accordion(
            children=[VBox(original_controlbox)],
            layout={"width":"120px",
                    "height":f"{self.h_upper_accordion}px", "margin":"0 0 0 20px", "padding":"1px"}
        )

        self.viewer1_accordion.set_title(0, "🔍 Viewer_1")
        self.viewer1_accordion.selected_index = None

        adjust_accordion_width(
            self.viewer1_accordion,
            closed_width="120px", opened_width=f"{w_v1}px",
            closed_height=f"{self.h_upper_accordion}px", opened_height=None,
        )

        # 4. viewer 2 (grid0)
        w_v2 = 645
        self.viewer2_accordion = Accordion(
            children=[grid0],
            layout={"width":"120px",
                    "height":f"{self.h_upper_accordion}px", "margin":"0 0 0 20px", "padding":"1px"}
        )

        self.viewer2_accordion.set_title(0, "🔍 Viewer_2")
        self.viewer2_accordion.selected_index = None

        adjust_accordion_width(
            self.viewer2_accordion,
            closed_width="120px", opened_width=f"{w_v2}px",
            closed_height=f"{self.h_upper_accordion}px", opened_height=None,
        )

        # Setting accordion to be closed when another accordiong is opened.
        upper_accordion = [self.setcalc_accordion, self.function_accordion,
                           self.viewer1_accordion, self.viewer2_accordion]
        setup_accordion_sync(accordions=upper_accordion)
        
        ###################################################################

        # --- Register callback ---
        self.v.observe(self._on_picked_changed_set_atoms, names=["picked"])
    #####################################
        self.v.observe(self._on_frame_changed_update_atoms, names='frame')
        
    def _on_frame_changed_update_atoms(self, change):
        """viewerのframeを手で動かして変化したら実行される関数。self.atomsを更新。
        """
        frame_idx = change['new']
        if isinstance(self.all_atoms, list) and 0 <= frame_idx < len(self.all_atoms):
            self.atoms = self.all_atoms[frame_idx]

        self.undo_frame_list[self.frame_num] = self.undo_list
        self.undo_list = self.undo_frame_list[self.v.frame]  # change corresponding undo_list

        self.redo_frame_list[self.frame_num] = self.redo_list
        self.redo_list = self.redo_frame_list[self.v.frame]  # change corresponding redo_list

        self.frame_num = self.v.frame
        
        self.update_display()

    ###################################

    def display(self):
        """display
        """
        top_widgets = HBox([self.setcalc_accordion, self.viewer1_accordion,
                            self.viewer2_accordion, self.function_accordion,
                           ])

        # 関数の下にouput_areaがあるバージョン
        # middle_widgets = HBox([self.read_write_accordion,
        #                        VBox([self.v, self.grid_cb],
        #                             layout=Layout(flex='0 0 auto')),
        #                        VBox([self.vh.control_box, self.output_area],
        #                             layout=Layout(flex='0 0 auto')),
        #                       ],
        #                       layout=Layout(align_items='flex-start')
        #                      )

        # h_grid_cb = 100
        # middle_widgets = HBox([self.read_write_accordion,
        #                        VBox([HBox([VBox([self.v, self.grid_cb],
        #                                         layout=Layout(min_width=f"{self.w}px", min_height=f"{self.h + h_grid_cb}px", flex="0 0 auto", overflow="visible", height="auto")
        #                                        ),
        #                                    VBox([self.vh.control_box],
        #                                         layout=Layout(min_width="495px", flex="0 0 auto")
        #                                        )
        #                                   ]),
        #                              self.output_area], layout=Layout(height=f"{self.h + self.output_height}px", overflow="auto", flex="0 0 auto")),
        #                       ], layout=Layout(overflow="auto", width=f"{self.w_left + self.w + 510}px"))

        middle_widgets = HBox([self.read_write_accordion,
                               VBox([HBox([VBox([self.v, self.grid_cb],
                                                layout=Layout(min_width=f"{self.w}px", flex="1 1 auto")
                                               ),
                                           VBox([self.vh.control_box],
                                                layout=Layout(min_width="495px", flex="0 0 auto")
                                               )
                                          ]),
                                     self.output_area]),
                              ], layout=Layout(overflow="auto", width=f"{self.w_left + self.w + 510}px"))
     
        all_widgets = VBox([top_widgets, middle_widgets])
        display(all_widgets)

    # def on_read_write_accordion_change(self, change):
    #     if change['name'] == 'selected_index':
    #         if change['new'] is not None:
    #             # Accordionが開いたとき → control_boxを非表示
    #             self.vh.control_box.layout.display = "none"
    #             self.output_area.layout.display = "none"

    #         else:
    #             # Accordionが閉じたとき → control_boxを表示
    #             self.vh.control_box.layout.display = None
    #             self.output_area.layout.display = None

    def set_representation(self, bcolor: str = "white",
                           unitcell: bool = False):
        # self.v.background = bcolor
        self.v.background = self.background_color
        self.struct = get_struct(self.atoms)
        self.v.add_representation(repr_type="ball+stick")
        if unitcell:
            self.v.add_representation(repr_type="unitcell")
        self.v.control.spin([0, 1, 0], pi * 1.1)
        self.v.control.spin([1, 0, 0], -pi * 0.45)
        # It's necessary to update indices of atoms \
        # specified by `get_struct` method.
        self.v._remote_call("replaceStructure", target="Widget",
                            args=self.struct)

    # def get_struct(self, atoms: Atoms, ext="pdb") -> List[Dict]:
    #     # For backward compatibility...
    #     return get_struct(atoms, ext=ext)

    def cal_nnp(self):
        mforce = (((self.atoms.get_forces()) ** 2).sum(axis=1).max()) ** 0.5
        pot = (
            self.atoms.get_potential_energy()
        )  # Faster to calculate energy after force.
        self.pot = pot
        self.mforce = mforce
        self.vh.nnptext.value = (f"pot energy : {pot:.2f} eV\nmax force  : " +
                                 f"{mforce:.4f} eV/A\ncalculator : " +
                                 f"{self.model_version} \n " +
                                 f"{' '*11} {self.calc_mode}")
        self.pots += [pot]
        self.traj += [self.atoms.copy()]

    def _update_Q(self):
        # Update `var atoms_pos` inside javascript.
        atoms = self.atoms
        if atoms.get_pbc().any():
            _, Q = atoms.cell.standard_form()
        else:
            Q = np.eye(3)
        Q_str = str(Q.tolist())
        var_str = f"this._Q = {Q_str}"
        self.v._execute_js_code(var_str)

    def _update_structure(self):
        struct = get_struct(self.atoms)
        self.struct = struct
        self.v._remote_call("replaceStructure", target="Widget", args=struct)
        self._update_Q()

    def update_display(self, clicked_button: Optional[Button] = None):
        # Force must be cleared before updating structure...
        self.v._remove_representations_by_name("const")  # It doesn't work if this line in show_const_checkbox below.
        self.clear_force()
        self.clear_axes()
        self._update_structure()
        self.cal_nnp()
        if self.show_force:
            self.add_force()
        if self.show_axes:
            self.add_axes()
        if self.vh.show_constraints_checkbox.value:
            self.show_constraints()

    def set_atoms(self, slider: Optional[FloatSlider] = None):
        """Allows specifying on each of the x, y, and z axes.
           If Only selected elements is set to True,
            allow index retrieval for only those elements
        """
        # If Only selected elements is set to False,
        #  get the indices of all atoms within the specified xyz range
        if (
            self.vh.display_only_selected_elements.value
        ) != 'Only selected elements':
            smols = [
                i for i, atom in enumerate(self.atoms)
                if (self.vh.setatoms_x.value[0] <= atom.x <=
                    self.vh.setatoms_x.value[1]) &
                   (self.vh.setatoms_y.value[0] <= atom.y <=
                    self.vh.setatoms_y.value[1]) &
                   (self.vh.setatoms_z.value[0] <= atom.z <=
                    self.vh.setatoms_z.value[1])
            ]
            # ↑ Since FloatRangeSlider is given in the form of either
            #    a tuple or list as [min, max], the minimum value is
            #    obtained with [0] and the maximum value with [1]

        # If Only selected elements is set to True, get only the indices of
        #  the specified elements within the specified xyz range
        else:
            try:
                selected_elements = (self.vh.selected_elements_textarea.
                                     value.split(","))
                selected_elements_list = [
                    str(_word.strip()) for _word in selected_elements
                    if _word.strip() != ""
                ]

                atoms_chemical_symbols = self.atoms.get_chemical_symbols()

            except Exception:
                print("None in selected elements")

            smols = [
                i for i, atom in enumerate(self.atoms)
                if (self.vh.setatoms_x.value[0] <= atom.x <=
                    self.vh.setatoms_x.value[1]) &
                   (self.vh.setatoms_y.value[0] <= atom.y
                    <= self.vh.setatoms_y.value[1]) &
                   (self.vh.setatoms_z.value[0] <= atom.z
                    <= self.vh.setatoms_z.value[1]) &
                   (atoms_chemical_symbols[i] in selected_elements_list)
            ]
            # ↑ Since FloatRangeSlider is given in the form of either a tuple
            #    or list as [min, max], the minimum value is obtained with [0]
            #    and the maximum value with [1]

        self.vh.selected_atoms_textarea.value = ", ".join(map(str, smols))

        # Add labels to atoms in Selected atoms only
        #  if the checkbox for "show index in Selected atoms" is checked
        self.show_one_part_index()



    def get_selected_atom_indices(self) -> List[int]:
        try:
            selected_atom_indices = (self.vh.selected_atoms_textarea.
                                     value.split(","))
            selected_atom_indices = [
                int(a.strip()) for a in selected_atom_indices
                if a.strip() != ""
            ]
            return selected_atom_indices
        except Exception:
            with self.vh.out_widget:
                print(traceback.format_exc(), file=sys.stderr)
            # `append_stderr` method shows same text twice somehow...
            # self.gui.out_widget.append_stderr(str(e))
        return []

    def _on_picked_changed_set_atoms(self, change: Bunch):
        # print(type(change), change)  # It has "name", "old", "new" keys.
        selected_atom_indices = self.get_selected_atom_indices()
        # Ex. picked format
        # {'atom1': {'index': 15, 'residueIndex': 0, 'resname': 'MOL', ...
        #  'name': '[MOL]1:A.FE'}, 'component': 0}
        index: int = self.v.picked.get("atom1", {}).get("index", -1)
        if index != -1:
            selected_atom_indices.append(index)
        # else:
        #     print(f"[ERROR] Unexpected format: v.picked {self.v.picked}")
        #  ↑ Removed because it appeared frequently

        selected_atom_indices = list(sorted(set(selected_atom_indices)))
        self.vh.selected_atoms_textarea.value = ", ".join(
            map(str, selected_atom_indices)
        )
        self.show_one_part_index()

    def rotate_view(self, clicked_button: Button):
        if clicked_button is self.vh.xplusview:
            self.v.control.rotate([-0.5, 0.5, 0.5, 0.5])
        elif clicked_button is self.vh.yplusview:
            self.v.control.rotate([-1 / sqrt(2), 0, 0, 1 / sqrt(2)])
        elif clicked_button is self.vh.zplusview:
            self.v.control.rotate([0, 1, 0, 0])
        elif clicked_button is self.vh.xminusview:
            self.v.control.rotate([-0.5, -0.5, -0.5, 0.5])
        elif clicked_button is self.vh.yminusview:
            self.v.control.rotate([0, 1 / sqrt(2), 1 / sqrt(2), 0])
        elif clicked_button is self.vh.zminusview:
            self.v.control.rotate([0, 0, 0, 1])
        else:
            raise ValueError("Unexpected button", clicked_button.description)

    # check #################################################################
    def reset_view(self, clicked_button: Button):
        tmp_angle = [1, 0, 0, 0, 0,
                     1, 0, 0, 0, 0,
                     1, 0, 0, 0, 0,
                     1]
        self.v.control.orient(tmp_angle)
        self.v.center()

    def append_tmpatoms_to_atomslist(self, a_atoms):
        """Function to save the current atoms type to atoms_list
            (not related to buttons, used for undo and redo buttons)
        """
        append_atoms = a_atoms.copy()
        self.undo_list.append(append_atoms)
        self.estimator = Estimator(calc_mode=self.calc_mode,
                                   model_version=self.model_version)
        self.undo_list[-1].calc = ASECalculator(self.estimator)
        self.redo_list = []

    def redo(self, clicked_button: Button):
        try:
            next_atoms = self.redo_list.pop()
            cell_size = next_atoms.get_cell()
            cell_pbc = next_atoms.pbc
            atoms_constraints = next_atoms.constraints

            # append tmp atoms to undo_list
            self.undo_list.append(self.atoms.copy())
            self.estimator = Estimator(calc_mode=self.calc_mode,
                                       model_version=self.model_version)
            self.undo_list[-1].calc = ASECalculator(self.estimator)

        except Exception:
            self.output_area.clear_output()
            with self.output_area:
                print("Can't Redo")
        else:
            # Match the number of atoms in the current atoms with the
            #  previous operation (If the number of atoms in the previous
            #  version is greater, append hydrogen atoms to the current
            #  version; if it's smaller, use pop to force the numbers to match)
            if len(self.atoms) < len(next_atoms):
                hydrogen_atom = Atom(symbol="H", position=[0, 0, 0])
                for i in range(len(next_atoms)-len(self.atoms)):
                    self.atoms.append(hydrogen_atom)
            elif len(self.atoms) > len(next_atoms):
                for i in range(len(self.atoms)-len(next_atoms)):
                    self.atoms.pop()
            else:
                pass

            self.atoms.symbols = next_atoms.get_chemical_symbols()
            self.atoms.positions = next_atoms.get_positions()
            self.atoms.cell = cell_size
            self.atoms.pbc = cell_pbc
            self.atoms.constraints = atoms_constraints

        self.update_display()

    def undo(self, clicked_button: Button):
        try:
            prev_atoms = self.undo_list.pop()
            cell_size = prev_atoms.get_cell()
            cell_pbc = prev_atoms.pbc
            atoms_constraints = prev_atoms.constraints

            # append tmp atoms to redo_list
            self.redo_list.append(self.atoms.copy())
            self.estimator = Estimator(calc_mode=self.calc_mode,
                                       model_version=self.model_version)
            self.redo_list[-1].calc = ASECalculator(self.estimator)

        except Exception:
            self.output_area.clear_output()
            with self.output_area:
                print("Can't Undo")
        else:
            # Match the number of atoms in the current atoms with the
            #  next operation (If the number of atoms in the next
            #  version is greater, append hydrogen atoms to the current
            #  version; if it's smaller, use pop to force the numbers to match)
            if len(self.atoms) < len(prev_atoms):
                hydrogen_atom = Atom(symbol="H", position=[0, 0, 0])
                for i in range(len(prev_atoms)-len(self.atoms)):
                    self.atoms.append(hydrogen_atom)
            elif len(self.atoms) > len(prev_atoms):
                for i in range(len(self.atoms)-len(prev_atoms)):
                    self.atoms.pop()
            else:
                pass

            self.atoms.symbols = prev_atoms.get_chemical_symbols()
            self.atoms.positions = prev_atoms.get_positions()
            self.atoms.cell = cell_size
            self.atoms.pbc = cell_pbc
            self.atoms.constraints = atoms_constraints

        self.update_display()

    def clear_textarea(self, clicked_button: Button):
        self.vh.selected_atoms_textarea.value = ""

        # Add labels to atoms in Selected atoms only if the checkbox
        #  for "show index in Selected atoms" is checked
        self.show_one_part_index()

    ##########################################################################

    def show_force_event(self, event: Bunch):
        self.show_force = self.vh.show_force_checkbox.value
        self.update_display()

    def show_axes_event(self, event: Bunch):
        self.show_axes = self.vh.show_axes_checkbox.value
        self.update_display()

    def show_constraints_event(self, event: Bunch):
        if event['name'] != 'value':
            return  # 'value'以外の変更には反応しない：3回呼ばれるのを防ぐ

        # whether the checkbox is true or false
        if self.vh.show_constraints_checkbox.value is True:
            self.update_display()  # update_display contains show_constraints function.
        else:
            self.v._remove_representations_by_name("const")
        
        self.update_display()

    def show_constraints(self):
        # whether atoms has constraints or not
        if self.atoms.constraints==[]:
            const_list = []
        else:
            const_list = self.atoms.constraints[0].index
            selection_str = f'@{(",").join([str(i) for i in const_list])}'
            self.v.add_representation(repr_type="ball+stick",
                                      selection=selection_str,
                                      color="royalblue",
                                      radius=0.5,
                                      name="const"
                                     )

    def show_index_event(self, event: Bunch):
        """'self.vh.show_index' adds indices to all atoms,
           'self.vh.show_index_one_part' adds indices only in Selected Atoms
        """
        # Branch based on the checkbox
        # (The if statement on this line is also included inside the
        #  show_one_part_index function, so it is unnecessary here,
        #  but it is included for clarity
        #  (the elif statement below is necessary))
        if event['name'] != 'value':
            return  # 'value'以外の変更には反応しない：3回呼ばれるのを防ぐ

        if self.vh.show_index_one_part_checkbox.value is True:
            self.show_one_part_index()
        elif ((self.vh.show_index_one_part_checkbox.value is False) &
              (self.vh.show_index_checkbox.value is True)):
            self.v.remove_label()
            self.show_all_index()
        else:
            self.v.remove_label()

        self.update_display()

    def show_one_part_index(self):
        """Add labels to atoms in Selected atoms
            only if the checkbox for 'show index in Selected atoms' is checked.
        """
        atom_indices = self.get_selected_atom_indices()
        label_indices = []
        for i in range(len(self.atoms)):
            if i in atom_indices:
                label_indices.append(i)
            else:
                label_indices.append("")

        label_indices = [str(i) for i in label_indices]
        if self.vh.show_index_one_part_checkbox.value is True:
            self.v.remove_label()
            self.v.add_label(
                color="blue", labelType="text",
                labelText=label_indices,
                zOffset=1.0, attachment='middle_center', radius=1.0
            )

    def show_all_index(self):
        """Visualize the indices of all atoms.
           The next 4 lines retrieve atoms within Selected Atoms
        """
        self.v.add_label(
            color="black", labelType="text",
            
            labelText=[str(i) for i in
                       range(self.atoms.get_global_number_of_atoms())],
            zOffset=1.0, attachment='middle_center', radius=1.0
        )

    # def on_load(self, clicked_button: Button):
        # self.output_area.clear_output()
        # try:
        #     load_atoms = read(self.uploader.current_dir,":")

        # except Exception:
        #     with self.output_area:
        #         print(f'"{self.uploader.current_dir}" is not atoms file.')

        # else:
        #     self.append_tmpatoms_to_atomslist(self.atoms)

        #     self.all_atoms = load_atoms
        #     for i in self.all_atoms:
        #         self.estimator = Estimator(calc_mode=self.calc_mode,
        #                                    model_version=self.model_version)
        #         i.calc = ASECalculator(self.estimator)

        #     self.atoms = self.all_atoms[0]

    def on_load(self, clicked_button: Button):
        self.output_area.clear_output()
        try:
            # 複数のAtomsを読み込む（trajectory対応）
            load_atoms = read(self.uploader.current_dir, ":")
        except Exception:
            with self.output_area:
                print(f'"{self.uploader.current_dir}" is not atoms file.')
        else:
            # 現在のatomsをundoリストに追加
            self.append_tmpatoms_to_atomslist(self.atoms)

            # frame数取得し、viewerに反映
            if type(load_atoms)==list:
                tmp_max_frame = len(load_atoms)-1
                
                # read(Atoms)をしてframe数が多くなる場合はundo,redoのframe数も増やす
                if len(self.undo_frame_list) < len(load_atoms):
                    for i in range(len(load_atoms)):
                        if i not in self.undo_frame_list:
                            self.undo_frame_list[i]=[]
                            self.redo_frame_list[i]=[]

            # Atomsのframe数が1の場合
            else:
                tmp_max_frame = 0

            self.v.max_frame = tmp_max_frame
                
            # all_atomsに読み込んだAtomsを格納
            self.all_atoms = load_atoms
    
            # 各Atomsにcalculatorを設定
            if isinstance(self.all_atoms, list):
                for i in self.all_atoms:
                    self.estimator = Estimator(calc_mode=self.calc_mode,
                                               model_version=self.model_version)
                    i.calc = ASECalculator(self.estimator)
                self.atoms = self.all_atoms[0]
            else:
                self.estimator = Estimator(calc_mode=self.calc_mode,
                                           model_version=self.model_version)
                self.all_atoms.calc = ASECalculator(self.estimator)
                self.atoms = self.all_atoms
    
            # NGLViewerに全フレームをロード
            self.v._remote_call("loadFrames", target="Widget", args=[{"structure": get_struct(self.all_atoms)}])
            
        self.update_display()
        self.v.center()  # bring the camera to the center
        self.selector.value=0  # navigate to the HOME screen.

    def on_write(self, clicked_button: Button):
        # self.output_area.clear_output()

        # AddEditorの画面でwriteしたときに、self.result.atomsをself.atomsに移行
        # To reflect change of viewer by AddEditor (from viewer to (viewer,nonviewer))
        if self.v.layout.display == "none":
            try:
                # viewerのatomsが複数の場合と一つの場合
                if type(self.result.atoms)==list:  
                    self.all_atoms = self.result.atoms
                else:
                    self.atoms = self.result.atoms  # これいれないとAddEditorで編集したものがself.atomsに反映されない
                    self.all_atoms[self.v.frame] = self.atoms
            except Exception:
                pass

        # If a file with the same name exists, save it with a number like _1, _2, etc.
        with self.output_area:
            counter = 1
            base, ext = os.path.splitext(self.write_file_text.value)
            new_filename = self.write_file_text.value

            # save_directory refer to dropdown folder
            if self.uploader.value_is_directory is True:
                save_dir = self.uploader.current_dir
            else:
                save_dir = ("/").join(self.uploader.current_dir.split("/")[:-1])

            # If a file with the same name exists, save it with a number.
            while os.path.exists(f"{save_dir}/{new_filename}"):
                new_filename = f"{base}_{counter}{ext}"
                counter += 1
                
            write(f"{save_dir}/{new_filename}", self.all_atoms)
            print(f"Atoms written to \n'{save_dir}/{new_filename}'")

        self.uploader.reload()

    def on_reload_uploader(self, clicked_button: Button):
        # uploader
        self.uploader.reload()

        # dropdown of func
        try:
            for tmp_dropdown in self.dropdown_func_list:
                tmp_dropdown.reload()
        except Exception:
            pass

    # def on_copy_frame(self, clicked_button: Button):
    #     """ insertしたatomsオブジェクトが同一のオブジェクトになってしまう...そもそも複雑な機能になるので消去
    #     """
    #     # prepare
    #     insert_index = self.v.frame
    #     inserted_atoms = self.atoms.copy()
    #     self.estimator = Estimator(calc_mode=self.calc_mode,
    #                                model_version=self.model_version)
    #     inserted_atoms.calc = ASECalculator(self.estimator)

    #     # insert
    #     self.all_atoms.insert(self.v.frame, inserted_atoms)
    #     self.v.max_frame += 1

    #     with self.output_area:
    #         self.output_area.clear_output()
    #         print(f"Copy Frame:{self.v.frame} to Frame:{self.v.frame+1}")

    #     # update undo list
    #     for key in sorted(self.undo_frame_list.keys(), reverse=True):
    #         if key >=insert_index:
    #             self.undo_frame_list[key+1] = self.undo_frame_list[key]
    #     self.undo_frame_list[insert_index] = []

    #     # update redo list
    #     for key in sorted(self.redo_frame_list.keys(), reverse=True):
    #         if key >=insert_index:
    #             self.redo_frame_list[key+1] = self.redo_frame_list[key]
    #     self.redo_frame_list[insert_index] = []

    #     # NGLViewerに全フレームをロード
    #     self.v._remote_call("loadFrames", target="Widget", args=[{"structure": get_struct(self.all_atoms)}])
    #     self.update_display()

    def on_calculator_change(self, selector_change):

        # set calculator for self.atoms
        self.estimator = Estimator(calc_mode=self.vh.setcalc_calculator.value,
                                   model_version=self.vh.setcalc_version.value)
        self.atoms.calc = ASECalculator(self.estimator)

        # set calculator for self.all_atoms
        for tmp_atoms in self.all_atoms:
            self.estimator = Estimator(calc_mode=self.vh.setcalc_calculator.value,
                                       model_version=self.vh.setcalc_version.value)
            tmp_atoms.calc = ASECalculator(self.estimator)

        try:
            self.update_display()  # try文をこれだけにしないと、try文なのにエラーが出る

        except:  # calculatorでエラーが起こったら元のcalculatorに設定しなおす
            # reset calculator for self.atoms
            self.estimator = Estimator(calc_mode=self.calc_mode,
                                       model_version=self.model_version)
            self.atoms.calc = ASECalculator(self.estimator)
    
            # reset calculator for self.all_atoms
            for tmp_atoms in self.all_atoms:
                self.estimator = Estimator(calc_mode=self.calc_mode,
                                           model_version=self.model_version)
                tmp_atoms.calc = ASECalculator(self.estimator)

            self.vh.setcalc_calculator.value = self.calc_mode
            self.vh.setcalc_version.value = self.model_version

        else:
            # get calculation condition
            self.calc_mode = self.vh.setcalc_calculator.value
            self.model_version = self.vh.setcalc_version.value
            with self.output_area:
                self.output_area.clear_output()
                print(f"Set calculator {self.calc_mode}, {self.model_version}")  # r2scan/v3.0.0とかだとupdate_displayでエラーがでるから後ろにしてる

        self.update_display()

    def on_selector_change(self, selector_change):
        """when the dropdown changes, change to the corresponding children
        """
        tmp_function_name = self.function_names[self.selector.value]  # ex).myopt
        # self.accordion_viewer_setting.selected_index=None
        
        # To reflect change of viewer by AddEditor (from viewer to (viewer,nonviewer))
        if self.v.layout.display == "none":
            try:
                # viewerのatomsが複数の場合と一つの場合
                if type(self.result.atoms)==list:  
                    self.all_atoms = self.result.atoms
                else:
                    self.atoms = self.result.atoms  # これいれないとAddEditorで編集したものがself.atomsに反映されない
                    self.all_atoms[self.v.frame] = self.atoms

            except Exception:  # 関数がグラフの時とか
                pass

            else:
                # CustomGUIのundo,redoとAddEditorのundo,redoをつなげる
                try:
                    self.undo_list = self.result.undo_list
                    self.redo_list = self.result.redo_list
                except Exception :
                    pass

                if type(self.result.atoms)==list:
                    for i in self.all_atoms:
                        self.estimator = Estimator(calc_mode=self.calc_mode,
                                                   model_version=self.model_version)
                        i.calc = ASECalculator(self.estimator)

                    self.atoms = self.all_atoms[0]

                    # self.frame_num = self.result.v.frame
                    
                else:
                    self.estimator = Estimator(calc_mode=self.calc_mode,
                                               model_version=self.model_version)
                    self.atoms.calc = ASECalculator(self.estimator)

                    self.all_atoms[self.v.frame] = self.atoms

            # frameの取り方がviewerによって異なるため
            try:
                self.v.frame = self.result.v.frame
            except:
                pass
                # passで問題ありそうならexceptの中身を以下に変更
                # try:
                #     self.v.frame = self.result.frame  # lotsatoms_viewはこっちが実行
                # except:
                #     pass

            # viewerのangleを共有（なくても不具合はないのでtry文に）
            try:
                tmp_angle = self.result.v.get_state()["_camera_orientation"]
                self.v.control.orient(tmp_angle)
            except:
                pass
            
            self.output_area.clear_output()
            self.update_display()

        # function is viewer or nonviewer (from (viewer,nonviewer) to viewer)
        if tmp_function_name in self.viewer_functions_list:
            selected_index = selector_change['new']
            tmp_function_name = self.function_names[selected_index]
            self.output_area.clear_output()

            self.v.layout.display = 'none'
            self.vh.control_box.layout.display = 'none'
            self.grid_cb.layout.display = 'none'
            with self.output_area:
                display(self.function_panels[selected_index].children[1])  # 関数名とsource codeボタンの表示

                viewer_func = getattr(self.utils, tmp_function_name)

                # viewerによってdisplayのさせ方が結構変わるので、どうしてもtry文が多くなってしまう...
                try:
                    self.result = viewer_func(self.all_atoms)  # AddEditorやlotsatomsviewer以外のviewerは全frameをview
                except:
                    self.result = viewer_func(self.atoms)  # AddEditorは現在のframeのAtomsだけを入力させる
                    if hasattr(self.result, "display"):
                        display(self.result.display())
                    else:
                        display(self.result)
                else:
                    display(self.result)
                
                # AddEditorのundo,redoとCustomGUIのundo,redoをつなげる
                try:
                    self.result.undo_list = self.undo_list
                    self.result.redo_list = self.redo_list
                except:
                    pass

            # viewerのangleを共有（なくても不具合はないのでtry文に）
            try:
                tmp_angle = self.v.get_state()["_camera_orientation"]
                self.result.v.control.orient(tmp_angle)
            except:
                pass
            
            self.output_area.layout = widgets.Layout(border='1px solid black')

        else:
            # redisplay
            self.v.layout.display = None
            self.vh.control_box.layout.display = None
            self.grid_cb.layout.display = None
            # self.output_area.clear_output()
            self.output_area.layout = widgets.Layout(border='1px solid black',
                                                     height=f"{self.output_height}px",
                                                     overflow='auto',
                                                     width=f'{self.output_width}px',
                                                     flex="0 0 auto")
            
            r = [HBox([HTML(value="<font size='2'> &nbsp; &nbsp; &nbsp; &nbsp; \
                                   &nbsp; &nbsp; NNP text: </font>"),
                       self.vh.nnptext, self.vh.undo, self.vh.redo]),
                 HBox([HTML(value="<font size='2'> &nbsp; &nbsp; </font>"),
                       self.vh.setatoms_z]),
                 HBox([HTML(value="<font size='2'> &nbsp; </font>"),
                       self.selected_atoms_hbox, self.vh.cleartextarea])
                ]      
            selected_selector_idx = selector_change["new"]
            r += [self.function_panels[selected_selector_idx]]
            self.vh.control_box.children = tuple(r)

            self.function_panels[selected_selector_idx].children[-1].selected_index = None  # background_jobのaccordionをオフ

        # uploaderとそのほかのドロップダウンを同期させるコードだが、少なからず重くなるので現時点ではコメントアウト
        # # Synchronize the all uploader. However, files that have been selected are exceptions.
        # for tmp_browser in self.dropdown_func_list:
        #     if tmp_browser.value_is_directory is True:
        #         tmp_browser.current_dir=self.uploader.current_dir
        #         tmp_browser.update_dropdown()

    def excute_utils_function(self,clicked_button: Button):
        """excute function in utils.py when put button.
        """
        self.append_tmpatoms_to_atomslist(self.atoms)
        with self.output_area:
            self.output_area.clear_output()
            tmp_function_name = self.function_names[self.selector.value]  # ex).myopt
            tmp_func = getattr(self.utils, tmp_function_name)

            # obtain specified argument in children
            tmp_children = self.function_panels[self.selector.value].children
            arg_values = {}
            for tmp_widget in tmp_children:

                # if dropdown(=atoms), read atoms
                # if isinstance(tmp_widget, widgets.Dropdown):
                if isinstance(tmp_widget, widgets.HBox):  # dropdownとリロードボタンのHBoxにしたのでtypeはHBoxになる。それにあわせて、tmp_widget⇒tmp_widget.children[0]に変更している
                    if isinstance(tmp_widget.children[0], widgets.Dropdown):
                        try:
                            argument_atoms = read(tmp_widget.children[0].value)
                        except:
                            raise ValueError("Atoms file may not be specified.")

                        self.estimator = Estimator(calc_mode=self.calc_mode,
                                                   model_version=self.model_version)
                        argument_atoms.calc = ASECalculator(self.estimator)
    
                        arg_values[tmp_widget.children[0].description] = argument_atoms

                # if floattext or text
                elif isinstance(tmp_widget, widgets.FloatText) or isinstance(tmp_widget, widgets.Text) or isinstance(tmp_widget, widgets.Textarea):  # これは現状elseでいい気も floattextはもう使ってない気がする

                    if tmp_widget.value == "[D] atoms":
                        arg_values[tmp_widget.description] = self.atoms

                    elif tmp_widget.value == "[D] all_atoms":
                        arg_values[tmp_widget.description] = self.all_atoms

                    elif tmp_widget.value == "[D] calculator":
                        self.estimator = Estimator(calc_mode=self.calc_mode,
                                                   model_version=self.model_version)
                        arg_values[tmp_widget.description] = ASECalculator(self.estimator)

                    else:
                        raw_value = tmp_widget.value
                        try:
                            # list, dict, set
                            value = ast.literal_eval(raw_value)
    
                        except Exception:
                            # float or int or str(contains bool)
                            if isinstance(raw_value, float) and raw_value.is_integer():  # int
                                value = int(raw_value)
                            elif raw_value=="False" or raw_value=="True":  # bool
                                value = eval(raw_value)
                            else:
                                value = raw_value
                        # print(value,type(value))
                        arg_values[tmp_widget.description] = value

            # # 引数の型が合っているかどうかのチェック用
            # for i in list(arg_values.values()):
            #     print(type(i),":  ",i)

            # excute function
            # print("argument: ",arg_values)
            self.result_func = tmp_func(**arg_values)


            # 関数を実行したreturnのtypeで場合分け
            if self.result_func is None:
                pass

            elif type(self.result_func) == ase.Atoms:
                self.estimator = Estimator(calc_mode=self.calc_mode,
                                           model_version=self.model_version)
                self.result_func.calc = ASECalculator(self.estimator)

                self.atoms = self.result_func
                
                if type(self.all_atoms) == list:
                    self.all_atoms[self.v.frame] = self.atoms
                else:
                    self.all_atoms = self.atoms

            elif type(self.result_func) == list or type(self.result_func) == ase.io.trajectory.TrajectoryReader:
                # trajectory to atoms_list
                if type(self.result_func) == ase.io.trajectory.TrajectoryReader:
                    atoms_list = [i.copy() for i in self.result_func]
                else:
                    atoms_list = self.result_func
                
                # listの中身に全部calculatorが設定できたら（=全部Atomsだったら）をtryで判定
                try:
                    for i in atoms_list:
                        self.estimator = Estimator(calc_mode=self.calc_mode,
                                                   model_version=self.model_version)
                        i.calc = ASECalculator(self.estimator)
                        
                except:
                    pass
                else:
                    if len(atoms_list)<self.v.frame:  # もとのatomsの方がframeが多い場合はframeを0にする
                        self.v.frame = 0

                    self.all_atoms = atoms_list
                    self.atoms = self.all_atoms[self.v.frame]

                    tmp_max_frame = len(atoms_list)-1

                    # read(Atoms)をしてframe数が多くなる場合はundo,redoのframe数も増やす
                    if len(self.undo_frame_list) < len(atoms_list):
                        for i in range(len(atoms_list)):
                            if i not in self.undo_frame_list:
                                self.undo_frame_list[i]=[]
                                self.redo_frame_list[i]=[]

                    # NGLViewerに全フレームをロード
                    self.v._remote_call("loadFrames", target="Widget", args=[{"structure": get_struct(self.all_atoms)}])
                    self.v.max_frame = tmp_max_frame

            else:
                pass

            # print(f"\n{'#'*20}\nThe return also can be obtained by 'self.result_func'.\n{'#'*20}\n")

        self.update_display()

    def run_background_job(self,licked_button: Button):
        """バックグラウンドジョブを流すまでやりたかったが、notebookが未保存認識されエラーになるので、
           notebookをつくるところまでしか実行していない
        """
        tmp_bg_accordion = self.function_panels[self.selector.value].children[-1].children[0]  # 最後の0はaccordionの0番目の意味
        run_folder_name = tmp_bg_accordion.children[0].value
        run_ipynb_name = tmp_bg_accordion.children[1].children[0].value  # 最後の0はHBoxの最初を取り出すため
                
        with self.output_area:
            self.output_area.clear_output()

        # 0. make directory & write atoms
            os.makedirs(run_folder_name, exist_ok=True)
            print(f"● directory      : {run_folder_name}")
            
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        
            # To write atoms later
            if os.path.isfile(f"{run_folder_name}/mol.json"):
                mol_json = f"mol_{timestamp}.{self.bg_write_extension}"
                all_mol_json = f"all_mol_{timestamp}.{self.bg_write_extension}"
            else:
                mol_json = f"mol.{self.bg_write_extension}"
                all_mol_json = f"all_mol.{self.bg_write_extension}"


        # 1. Get argument (same as excute_utils_function) ####
            tmp_function_name = self.function_names[self.selector.value]  # ex).myopt
            tmp_func = getattr(self.utils, tmp_function_name)

            # obtain specified argument in children
            tmp_children = self.function_panels[self.selector.value].children
            arg_values = {}
            dropdown_code_lines = []
            mol_argument_list = []  # str in
            all_mol_argument_list = []  # str in
            calculator_argument_list = []  # str in

            count_appear_dropdown = 0
            count_appear_mol = 0
            count_appear_all_mol = 0
            
            for tmp_widget in tmp_children:
                # if dropdown(=atoms), read atoms
                # if isinstance(tmp_widget, widgets.Dropdown):
                if isinstance(tmp_widget, widgets.HBox):  # dropdownとリロードボタンのHBoxにしたのでtypeはHBoxになる。それにあわせて、tmp_widget⇒tmp_widget.children[0]に変更している
                    if isinstance(tmp_widget.children[0], widgets.Dropdown):
                        argument_name = f"mol_argument_gui{count_appear_dropdown}"
                        dropdown_code_lines.append(
                            f"\n{argument_name} = read('{tmp_widget.children[0].value}')"
                            f"\n{argument_name}.calc = calculator"
                        )

                        arg_values[tmp_widget.children[0].description] = argument_name
                        count_appear_dropdown += 1
                
                # if floattext or text
                elif isinstance(tmp_widget, widgets.FloatText) or isinstance(tmp_widget, widgets.Text):  # これは現状elseでいい気も

                    if tmp_widget.value == "[D] atoms":
                        arg_values[tmp_widget.description] = "mol"

                        if count_appear_mol == 0:
                            write(f"{run_folder_name}/{mol_json}", self.atoms)
                            print(f"● write atoms    : {mol_json}")
                            mol_argument_list.append(tmp_widget.description)

                        count_appear_mol += 1

                    elif tmp_widget.value == "[D] all_atoms":
                        arg_values[tmp_widget.description] = "all_mol"

                        if count_appear_all_mol == 0:
                            write(f"{run_folder_name}/{all_mol_json}", self.all_atoms)
                            print(f"● write all_atoms: {all_mol_json}")
                            all_mol_argument_list.append(tmp_widget.description)
                            
                        count_appear_all_mol += 1
                        
                    elif tmp_widget.value == "[D] calculator":
                        arg_values[tmp_widget.description] = "calculator"
                        calculator_argument_list.append(tmp_widget.description)

                    else:
                        raw_value = tmp_widget.value
                        try:
                            # list, dict, set
                            value = ast.literal_eval(raw_value)
    
                        except Exception:
                            # float or int or str(contains bool)
                            if isinstance(raw_value, float) and raw_value.is_integer():  # int
                                value = int(raw_value)
                            elif raw_value=="False" or raw_value=="True":  # bool
                                value = eval(raw_value)
                            else:
                                value = raw_value
                        arg_values[tmp_widget.description] = value    

        # 2. Extract the content to be written in notebook from utils.py
            # open utils.py
            with open(a.utils_path, "r", encoding="utf-8") as f:
                source = f.read()
            
            # Extract import & function statement
            parsed = ast.parse(source)
            
            import_lines = []
            function_lines = []
            all_lines = []
            source_lines = source.splitlines()
            
            for node in parsed.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_lines.append(source_lines[node.lineno - 1])
                elif isinstance(node, ast.FunctionDef):
                    start = node.lineno - 1
                    end = max([n.lineno for n in ast.walk(node) if hasattr(n, 'lineno')])
                    function_lines.extend(source_lines[start:end])
                    function_lines.append("")  # 関数間に空行を追加
            
            # Create import & fuction code
            import_code = "\n".join(import_lines)
            function_code = "\n".join(function_lines)
            all_lines.append(import_code)
            all_lines.append(function_code)

            # Create excute code
            # arg_values={"sn":10, "iter_count":False}

            # .1 mol code
            mol_code = f"""
# calculator
estimator = Estimator(calc_mode='{self.calc_mode}', model_version='{self.model_version}')
calculator = ASECalculator(estimator)
"""
            if count_appear_mol > 0:
                mol_code += f"""
# mol
mol = read('{mol_json}')
mol.calc = calculator
"""

            if count_appear_all_mol > 0:
                mol_code += f"""
# all_mol
all_mol = read('{all_mol_json}')
all_mol.calc = calculator
"""

            all_lines.append(mol_code)

            # .2 mol_argument code
            if count_appear_dropdown>0:
                dropdown_code_lines_str = "# mol argument"+"\n".join(dropdown_code_lines)
                all_lines.append(dropdown_code_lines_str)
            else:
                dropdown_code_lines_str = ""

            # .3 function code
            arg_str_parts = []
            for k, v in arg_values.items():
                if isinstance(v, str) and v.startswith("mol_argument_gui"):
                    arg_str_parts.append(f"{k}={v}")
                    # print(k,": dropdown")

                elif k in mol_argument_list or k in all_mol_argument_list:
                    arg_str_parts.append(f"{k}={v}")
                    # print(k,": mol")

                elif k in calculator_argument_list:
                    arg_str_parts.append(f"{k}={v}")
                    # print(k,": calc")

                else:
                    arg_str_parts.append(f"{k}={repr(v)}")
            
            arg_str = ", ".join(arg_str_parts)

            
            excute_code = f"""
# excute function
result = {self.function_names[self.selector.value]}({arg_str})

# save result if result is atoms
try:
    write('result_{mol_json}', result)
except Exception:
    pass
"""
            all_lines.append(excute_code)
            
        # 3. create notebook
            if os.path.isfile(f"{run_folder_name}/{run_ipynb_name}.ipynb"):
                run_ipynb_name = run_ipynb_name + "_" + timestamp

            create_utils_notebook(f"{run_folder_name}/{run_ipynb_name}.ipynb",
                                  all_lines)
            
            print(f"● create notebook: {run_ipynb_name}.ipynb")

        # 4. excute background job
            # バックグラウンドジョブ流すときにエラーが出るのでコメントアウト
            # print("\nStart background job...")
            # input_file = f"{run_folder_name}/{run_ipynb_name}.ipynb"
            # output_file = f"{run_folder_name}/bg_{run_ipynb_name}.ipynb"
            # !mtl-bg-job run "{input_file}" "{output_file}"
        

    def show_docstring(self,clicked_button: Button):
        with self.output_area:
            # viewerはoutput_areaに表示されるため、このif文を入れないとsource codeを出力したらviewerも消えてしまう
            if self.function_names[self.selector.value] not in self.viewer_functions_list:
                self.output_area.clear_output()

            tmp_function_name = self.function_names[self.selector.value]  # ex).myopt
            tmp_func = getattr(self.utils, tmp_function_name)
            # print(help(tmp_func))

            try:
                source_code = inspect.getsource(tmp_func)
            except OSError:
                print("Failed to retrieve source code. \
                       The function may be defined in a C extension or similar.")
            else:
                # get source_code
                formatter = HtmlFormatter(style="default", noclasses=True)
                highlighted_code = highlight(source_code, PythonLexer(), formatter)

                # display
                IPython.display.display(IPython.display.HTML(
                    f"<div style='margin-top:10px;'><b>--- Source code for \
                      '{tmp_function_name}' ---</b></div>"))
                IPython.display.display(IPython.display.HTML(highlighted_code))


class FileBrowser:
    def __init__(self,
                 start_path=".",
                 description="Browse:",
                 view_type="dropdown", # dropdown or select
                 w=450, h=200):

        self.allow_file_types = ("pdb", "c3xml", "cif", "cml", "dx",
                                 "gamess", "jdx", "jxyz", "magres",
                                 "mol", "json", "molden", "phonon", "sdf",
                                 "xodydata", "xsf", "xyz", "vasp", "traj"
                                )
        self.current_dir_component = os.getcwd().split("/")
        self.current_dir = ("/").join(self.current_dir_component)

        if view_type=="select":
            self.dropdown = Select(description=description, layout={"width":f"{w}px","height":f"{h}px"})
        else:
            self.dropdown = Dropdown(description=description, layout={"width":f"{w}px"})
            
        # self.load_button = Button(description="Load .xyz")

        self.dropdown.observe(self.on_select, names='value')
        # self.load_button.on_click(self.on_load)

        self.update_dropdown()
        self.value_is_directory = True

    def update_dropdown(self):
        try:
            items = os.listdir(self.current_dir)
        except PermissionError:
            items = []

        folders = [tmp_item for tmp_item in items if os.path.isdir(self.current_dir+"/"+tmp_item)]
        files = [tmp_item for tmp_item in items if os.path.isfile(self.current_dir+"/"+tmp_item)]
    
        # options = [("--Up--", os.path.dirname(("/").join(self.current_dir_component[:-1])))]  # ← value を親ディレクトリのパスに
        options = [(f"{self.current_dir}",self.current_dir)]
        options += [(".. /", ("/").join(self.current_dir_component[:-1]))]  # ← value を親ディレクトリのパスに
        options += [(f"📁 {f}", os.path.join(self.current_dir, f)) for f in sorted(folders)]
        options += [(f"📄 {f}", os.path.join(self.current_dir, f)) for f in sorted(files)]
    
        self.dropdown.options = options
        # self.dropdown.value = None
        self.dropdown.value = options[0][1]

    def on_select(self, change):
        selected_path = change["new"]
        if selected_path is None:
            return

        # イベント解除
        self.dropdown.unobserve(self.on_select, names='value')
        
        if os.path.isdir(selected_path):
            self.current_dir = selected_path
            self.current_dir_component = self.current_dir.split("/")
            self.update_dropdown()
            self.value_is_directory = True

        elif os.path.isfile(selected_path):
            # If file is atoms file, self.current_dir become atoms file.
            if selected_path.split("/")[-1].split(".")[-1].lower() in self.allow_file_types:
                self.current_dir = selected_path
                self.value_is_directory = False
            # else file is not atoms file, undo dropdown value
            else:
                self.dropdown.value = self.current_dir
        else:
            pass

        # イベント再登録
        self.dropdown.observe(self.on_select, names='value')

    def display(self):
        return VBox([self.dropdown])

    def reload(self):
        """ if a folder is selected, update that folder.
            if a file is selected, update the folder in the hierarchy.
        """
        # if file is selected
        if self.current_dir.split("/")[-1].split(".")[-1].lower() in self.allow_file_types:
            bef_update_file = self.current_dir
            self.current_dir = "/".join(self.current_dir.split("/")[:-1])
            self.dropdown.value = self.current_dir
            
            self.update_dropdown()
            
            # if the file can be selected in the dropdown
            try:
                self.dropdown.value = bef_update_file
                self.current_dir = bef_update_file
            except Exception:
                pass

        # if folder is selected
        else:
            self.update_dropdown()


def create_utils_notebook(title_ipynb: str, codes: list):
    """Create jupyter notebook

    Args:
        title_ipynb (str): title of created notebook
        codes (list): code to write notebook

    Returns:
        jupyter notebook
    """
    all_cell = [nbformat.v4.new_code_cell(code) for code in codes]
    notebook = nbformat.v4.new_notebook(cells=all_cell)

    with open(title_ipynb, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)
        f.flush()
        os.fsync(f.fileno())


def adjust_accordion_width(accordion, closed_width='200px', opened_width='600px', closed_height=None, opened_height=None):
    def _adjust_width(change):
        accordion.layout.width = closed_width if change['new'] is None else opened_width
        accordion.layout.height = closed_height if change['new'] is None else opened_height
    accordion.observe(_adjust_width, names='selected_index')


def setup_accordion_sync(accordions):
    """指定したaccordionについて、開いたら別のaccordionが閉じるようにする

    Returns:
        accordion.observe
    """
    def make_observer(current_idx):
        def observer(change):
            if change['name'] == 'selected_index':
                if change['new'] is not None:  # このifはaccordionが開かれた場合
                    for i, acc in enumerate(accordions):
                        if i != current_idx:
                            acc.selected_index = None
        return observer

    # 各Accordionにobserverを設定
    for idx, acc in enumerate(accordions):
        acc.observe(make_observer(idx), names='selected_index')


# Drag atoms, all_atoms and calculator
drag_html = """
<div style = "display: flex; gap: 10 px">
  <div draggable="true" ondragstart="event.dataTransfer.setData('text/plain', '[D] atoms')" 
       style="padding:0px; margin:1px; background-color:#add8e6; color:grey; width:80px; text-align:center; font-weight:bold;">
       atoms
  </div>
  <div draggable="true" ondragstart="event.dataTransfer.setData('text/plain', '[D] all_atoms')" 
       style="padding:0px; margin:1px; background-color:#90ee90; color:grey; width:80px; text-align:center; font-weight:bold;">
       all_atoms
  </div>
  <div draggable="true" ondragstart="event.dataTransfer.setData('text/plain', '[D] calculator')" 
       style="padding:0px; margin:1px; background-color:#ffcccb; color:grey; width:80px; text-align:center; font-weight:bold;">
       calculator
  </div>
</div>

<script>
  let interval = setInterval(() => {
    let input = document.querySelector('input[type="text"]');
    if (input) {
      input.addEventListener('dragover', function(event) {
        event.preventDefault();
      });
      input.addEventListener('drop', function(event) {
        event.preventDefault();
        const droppedText = event.dataTransfer.getData('text/plain');
        input.value = droppedText;  // dropしたテキストに完全に置き換え（なぜかうまくいかない...）
        input.dispatchEvent(new Event('input'));
      });
      clearInterval(interval);
    }
  }, 500);
</script>
"""