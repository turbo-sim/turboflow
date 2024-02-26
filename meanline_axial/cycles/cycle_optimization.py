import os
import time
import yaml
import copy
import datetime
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .. import solver
from .. import utilities
from .. import fluid_properties as props

from . import cycle_recuperated
from . import cycle_split_compression

COLORS_MATLAB = utilities.COLORS_MATLAB
LABEL_MAPPING = {
    "s": "Entropy [J/kg/K]",
    "T": "Temperature [K]",
    "p": "Pressure [Pa]",
    "h": "Enthalpy [J/kg]",
    "d": "Density [kg/m$^3$]",
    "a": "Speed of sound [m/s]",
    "Z": "Compressibility factor [-]",
    "heat": "Heat flow rate [W]",
}

# Cycle configurations available
CYCLE_CONFIGURATIONS = [
    "recuperated",
    "split_compression",
    "recompression",
]

TOPOLOGY_MAPPING = {
    CYCLE_CONFIGURATIONS[0]: cycle_recuperated.evaluate_cycle,
    CYCLE_CONFIGURATIONS[1]: cycle_split_compression.evaluate_cycle,
    CYCLE_CONFIGURATIONS[2]: cycle_split_compression.evaluate_cycle,
}

GRAPHICS_PLACEHOLDER = {
    "process_lines": {},
    "state_points": {},
    "pinch_point_lines": {},
}


class ThermodynamicCycleProblem(solver.OptimizationProblem):
    """
    A class to represent a thermodynamic cycle optimization problem.

    This class provides functionalities to load and update the configuration for
    the thermodynamic cycle, and to perform interactive plotting based on the current
    configuration.

    Attributes
    ----------
    plot_initialized : bool
        Flag to indicate if the plot has been initialized.
    constraints : dict
        Dictionary holding the constraints of the problem.
    fixed_parameters : dict
        Dictionary holding the fixed parameters of the problem.
    design_variables : dict
        Dictionary holding the current values of the design variables.
    lower_bounds : dict
        Dictionary holding the lower bounds of the design variables.
    upper_bounds : dict
        Dictionary holding the upper bounds of the design variables.
    keys : list
        List of keys (names) of the design variables.
    x0 : np.ndarray
        Initial guess for the optimization problem.

    Methods
    -------
    update_config(configuration):
        Update the problem's configuration based on the provided dictionary.
    load_config_from_file(configuration_file):
        Load and update the problem's configuration from a specified file.
    plot_cycle_interactive(configuration_file, update_interval=0.20):
        Perform interactive plotting, updating the plot based on the configuration file.
    """

    def __init__(self, configuration, out_dir=None):
        """
        Constructs all the necessary attributes for the ThermodynamicCycleProblem object.

        Parameters
        ----------
        config : dict
            Dictionary containing the configuration for the thermodynamic cycle problem.
        """
        # Initialize variables
        self.figure = None
        self.figure_TQ = None
        self.configuration = copy.deepcopy(configuration)
        self.update_configuration(self.configuration)
        self.graphics = copy.deepcopy(GRAPHICS_PLACEHOLDER)

        # Create fluid object to calculate states used for the scaling of design variables
        self.fluid = props.Fluid(**self.fixed_parameters["working_fluid"])
        self._calculate_special_points()

        # Define filename with unique date-time identifier
        if out_dir == None:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.out_dir = f"results/case_{current_time}"

        # Create a directory to save simulation results
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        # Save the initial configuration as YAML file
        config_data = {k: v for k, v in self.configuration.items()}
        config_data = utilities.convert_numpy_to_python(config_data, precision=12)
        config_file = os.path.join(self.out_dir, "initial_configuration.yaml")
        with open(config_file, "w") as file:
            yaml.dump(config_data, file, default_flow_style=False, sort_keys=False)

    def update_configuration(self, configuration):
        """
        Update the problem's configuration based on the provided dictionary.

        Parameters
        ----------
        config : dict
            Dictionary containing the new configuration for the thermodynamic cycle problem.
        """
        conf = configuration
        self.cycle_topology = conf["cycle_topology"]
        self.plot_settings = conf["plot_settings"]
        self.constraints = conf["constraints"]
        self.fixed_parameters = conf["fixed_parameters"]
        self.objective_function = conf["objective_function"]
        self.variables = {k: v["value"] for k, v in conf["design_variables"].items()}
        self.lower_bounds = {k: v["min"] for k, v in conf["design_variables"].items()}
        self.upper_bounds = {k: v["max"] for k, v in conf["design_variables"].items()}
        self.keys = list(self.variables.keys())
        self.x0 = np.asarray([var for var in self.variables.values()])

    def _calculate_special_points(self):
        """
        Calculates and stores key thermodynamic states for the fluid used in the cycle.

        This method computes two specific thermodynamic states:
        1. Saturated liquid state at the heat sink inlet temperature.
        2. Dilute gas state at the heat source inlet temperature.

        If the heat sink inlet temperature is below the fluid's critical temperature,
        the saturated liquid state is calculated at this temperature. If it is above
        the critical temperature, the state is calculated along the pseudocritical line.

        The dilute gas state is calculated at the heat source inlet temperature and at
        very low pressure (slightly above the triple pressure)

        The calculated states, along with the fluid's critical and triple point properties,
        are stored in the `fixed_parameters` dictionary under the 'fluid' key.

        These states are intended to define the bounds of the design variables specified
        in the YAML configuration file, for example:

            compressor_inlet_enthalpy:
            min: 0.9*$fluid.liquid_sink_temperature.h
            max: 2.0*$fluid.liquid_sink_temperature.h
            value: 0.2
            turbine_inlet_pressure:
            min: 0.75*$fluid.critical_point.p
            max: 5.00*$fluid.critical_point.p
            value: 0.5
            turbine_inlet_enthalpy:
            min: 1.10*$fluid.critical_point.h
            max: 1.00*$fluid.gas_source_temperature.h
            value: 0.90

        """
        # Compute saturated liquid state at sink temperature
        # Use pseudocritical line if heat sink is above critical temperature
        T_sink = self.fixed_parameters["heat_sink"]["inlet_temperature"]
        crit = self.fluid.critical_point
        if T_sink < crit.T:
            state_sat = self.fluid.set_state(props.QT_INPUTS, 0.0, T_sink)
        else:
            state_sat = self.fluid.set_state(props.DmassT_INPUTS, crit.rho, T_sink)

        # Compute dilute gas state at heat source temperature
        T_source = self.fixed_parameters["heat_source"]["inlet_temperature"]
        p_triple = 1.01 * self.fluid.triple_point_liquid.p
        state_dilute = self.fluid.set_state(props.PT_INPUTS, p_triple, T_source)

        # Save states in the fixed parameters dictionary
        self.fixed_parameters_bis = copy.deepcopy(self.fixed_parameters)
        self.fixed_parameters_bis["cycle_fluid"] = {
            "critical_point": self.fluid.critical_point.to_dict(),
            "triple_point_liquid": self.fluid.triple_point_liquid.to_dict(),
            "triple_point_vapor": self.fluid.triple_point_vapor.to_dict(),
            "liquid_at_heat_sink_temperature": state_sat.to_dict(),
            "gas_at_heat_source_temperature": state_dilute.to_dict(),
        }

    def load_configuration_file(self, config_file):
        """
        Load and update the problem's configuration from a specified file.

        Parameters
        ----------
        config_file : str
            Path to the configuration file.
        """
        config = utilities.read_configuration_file(config_file)
        self.update_configuration(config["problem_formulation"])
        self._calculate_special_points()

    def get_optimization_values(self, x):
        """
        Evaluate optimization problem
        """
        # Create dictionary from array of normalized design variables
        self.variables = dict(zip(self.keys, x))
        vars_physical = self._scale_normalized_to_physical(self.variables)

        # Evaluate thermodynamic cycle
        if self.cycle_topology in CYCLE_CONFIGURATIONS:
            self.cycle_data = TOPOLOGY_MAPPING[self.cycle_topology](
                vars_physical,
                self.fixed_parameters,
                self.constraints,
                self.objective_function,
            )
        else:
            options = ", ".join(f"'{k}'" for k in CYCLE_CONFIGURATIONS)
            raise ValueError(
                f"Invalid cycle topology: '{self.cycle_topology}'. Available options: {options}"
            )

        # Define objective function and constraints
        self.f = self.cycle_data["objective_function"]
        self.c_eq = self.cycle_data["equality_constraints"]
        self.c_ineq = self.cycle_data["inequality_constraints"]

        # Combine objective function and constraints
        out = self.merge_objective_and_constraints(self.f, self.c_eq, self.c_ineq)

        return out

    def get_bounds(self):
        bounds = []
        for _ in self.variables:
            bounds.append((0.00, 1.00))
        return bounds

    def get_n_eq(self):
        return self.get_number_of_constraints(self.c_eq)

    def get_n_ineq(self):
        return self.get_number_of_constraints(self.c_ineq)

    def _scale_normalized_to_physical(self, vars_normalized):
        # Define helper function
        def scale_value(min_val, max_val, normalized_val):
            return min_val + (max_val - min_val) * normalized_val

        # Loop over normalized variables
        vars_physical = {}
        for key, value in vars_normalized.items():
            try:
                # Fetch the bounds expressions
                lower_expr = self.lower_bounds[key]
                upper_expr = self.upper_bounds[key]

                # Try to evaluate bounds as expressions if they are strings
                if isinstance(lower_expr, str):
                    lower = utilities.render_and_evaluate(
                        lower_expr, self.fixed_parameters_bis
                    )
                else:
                    lower = lower_expr
                if isinstance(upper_expr, str):
                    upper = utilities.render_and_evaluate(
                        upper_expr, self.fixed_parameters_bis
                    )
                else:
                    upper = upper_expr

                vars_physical[key] = scale_value(lower, upper, value)

            except (KeyError, ValueError) as e:
                raise ValueError(f"Error processing bounds for '{key}': {e}")

        return vars_physical

    def to_excel(self, filename="performance.xlsx"):
        """
        Exports the cycle performance data to Excel file
        """

        # Define variable map
        variable_map = {
            "fluid_name": {"name": "fluid_name", "unit": "-"},
            "T": {"name": "temperature", "unit": "K"},
            "p": {"name": "pressure", "unit": "Pa"},
            "rho": {"name": "density", "unit": "kg/m3"},
            "Q": {"name": "quality", "unit": "-"},
            "Z": {"name": "compressibility_factor", "unit": "-"},
            "u": {"name": "internal_energy", "unit": "J/kg"},
            "h": {"name": "enthalpy", "unit": "J/kg"},
            "s": {"name": "entropy", "unit": "J/kg/K"},
            "cp": {"name": "isobaric_heat_capacity", "unit": "J/kg/K"},
            "cv": {"name": "isochoric_heat_capacity", "unit": "J/kg/K"},
            "gamma": {"name": "heat_capacity_ratio", "unit": "-"},
            "a": {"name": "speed_of_sound", "unit": "m/s"},
            "mu": {"name": "dynamic_viscosity", "unit": "Pa*s"},
            "k": {"name": "thermal_conductivity", "unit": "W/m/K"},
            "superheating": {"name": "superheating_degree", "unit": "K"},
            "subcooling": {"name": "subcooling_degree", "unit": "K"},
        }

        # Initialize a list to hold all rows of the DataFrame
        data_rows = []

        # Prepare the headers and units rows
        headers = ["state"]
        units_row = ["units"]

        for key in variable_map:
            headers.append(variable_map[key]["name"])
            units_row.append(variable_map[key]["unit"])

        # Iterate over each component in the dictionary
        for component_name, component in self.cycle_data["components"].items():
            if component["type"] == "heat_exchanger":
                # Handle heat exchanger sides separately
                for side in ["hot_side", "cold_side"]:
                    # Append the data for state_in and state_out to the rows list
                    state_in = component[side]["state_in"].to_dict()
                    state_out = component[side]["state_out"].to_dict()
                    data_rows.append(
                        [f"{component_name}_{side}_in"]
                        + [state_in.get(key, None) for key in variable_map]
                    )
                    data_rows.append(
                        [f"{component_name}_{side}_out"]
                        + [state_out.get(key, None) for key in variable_map]
                    )
            else:
                # Handle non-heat exchanger components
                # Append the data for state_in and state_out to the rows list
                state_in = component["state_in"].to_dict()
                state_out = component["state_out"].to_dict()
                data_rows.append(
                    [f"{component_name}_in"]
                    + [state_in.get(key, None) for key in variable_map]
                )
                data_rows.append(
                    [f"{component_name}_out"]
                    + [state_out.get(key, None) for key in variable_map]
                )

        # Create a DataFrame with data rows
        df = pd.DataFrame(data_rows, columns=headers)

        # Insert the units row
        df.loc[-1] = units_row  # Adding a row
        df.index = df.index + 1  # Shifting index
        df = df.sort_index()  # Sorting by index

        # # Export to Excel
        # df.to_excel(
        #     os.path.join(self.out_dir, filename),
        #     index=False,
        #     header=True,
        #     sheet_name="cycle_states",
        # )

        # Prepare energy_analysis data
        df_2 = pd.DataFrame(
            list(self.cycle_data["energy_analysis"].items()),
            columns=["Parameter", "Value"],
        )

        # Export to Excel
        filename = os.path.join(self.out_dir, filename)
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="cycle_states")
            df_2.to_excel(writer, index=False, sheet_name="energy_analysis")

    def plot_cycle(self):
        """
        Plots or updates the thermodynamic cycle diagrams based on current settings,
        including the option to include a pinch point diagram.

        This function is capable of both creating new cycle diagrams and updating existing ones.
        It's particularly useful in dynamic scenarios such as during optimization steps,
        where the plot needs to be refreshed continually with new data.
        The method also supports real-time updates based on the latest configuration settings.

        The function first determines the number of subplots required based on the cycle diagrams
        specified in the 'plot_settings' attribute and whether a pinch point diagram is included.
        It then either initializes a new figure and axes or updates existing ones.
        Each thermodynamic diagram (phase diagram and cycle components) and the optional pinch point
        diagram are plotted or updated accordingly.

        The method ensures that the plot reflects the current state of the cycle, including any
        changes during optimization or adjustments to configuration settings.

        The data of the plots can be updated interactively, but the subplot objects created can
        only be specified upon class initialization. For example, it is not possible to switch
        on and off the pinch point diagram or to add new thermodynamic diagrams. The class would
        have to be re-initialized upon those scenarios. The reason for this choice is that re-creating
        a figure would be too time consuming and not practical for the real-time updating of the plots

        """

        # Determine the number of subplots
        include_pinch_diagram = self.plot_settings.get("pinch_point_diagram", False)
        ncols = len(self.plot_settings["diagrams"]) + int(include_pinch_diagram)
        nrows = 1

        # Initialize the figure and axes
        if not (self.figure and plt.fignum_exists(self.figure.number)):
            # Reset the graphics objects if the figure was closed
            self.graphics = copy.deepcopy(GRAPHICS_PLACEHOLDER)
            self.figure, self.axes = plt.subplots(
                nrows, ncols, figsize=(5.2 * ncols, 4.8)
            )
            self.axes = utilities.ensure_iterable(self.axes)

        # Plot or update each thermodynamic_diagram
        for i, ax in enumerate(self.axes[:-1] if include_pinch_diagram else self.axes):
            plot_config = self.plot_settings["diagrams"][i]
            plot_config = self._get_diagram_default_settings(plot_config)
            self._plot_thermodynamic_diagram(ax, plot_config, i)

        # Plot or update the pinch point diagram
        if include_pinch_diagram:
            self._plot_pinch_point_diagram(self.axes[-1], ncols - 1)

        # Adjust layout and refresh plot
        self.figure.tight_layout(pad=2)
        plt.draw()
        plt.pause(0.05)

    def plot_cycle_realtime(self, configuration_file, update_interval=0.1):
        """
        Perform interactive plotting, updating the plot based on the configuration file.

        Parameters
        ----------
        config_file : str
            Path to the configuration file.
        update_interval : float, optional
            Time interval in seconds between plot updates (default is 0.1 seconds).

        """

        def wait_for_input():
            input()
            self.enter_pressed = True

        # Initialize secondary thread to wait for user input
        self.enter_pressed = False
        self.input_thread = threading.Thread(target=wait_for_input)
        self.input_thread.daemon = True
        self.input_thread.start()

        # Print instructions message
        print("-" * 80)
        print(" Creating thermodynamic cycle interactive plot")
        print("-" * 80)
        print(f" The current configuration file is: '{configuration_file}'")
        print(" Modify the configuration file and save it to update the plot")
        print(" Try to find a good initial guess for the optimization.")
        print(" Using a feasible initial guess improves optimization convergence.")
        print(" Press 'enter' to continue.")
        print("-" * 80)

        # Update the plot until termination signal
        while not self.enter_pressed:
            # Read the configuration file
            self.load_configuration_file(configuration_file)
            self.get_optimization_values(self.x0)
            self.plot_cycle()

            # Wait for the specified interval before updating again
            time.sleep(update_interval)

            # Exit interactive plotting when the user closes the figure
            if not plt.fignum_exists(self.figure.number):
                break

            # Exit interactive plotting when the user presses enter
            if self.enter_pressed:
                break

    def plot_cycle_callback(self, x, iter):
        """
        Plot and save the thermodynamic cycle at different optimization iterations

        This function is intended to be passed as callback argument to the optimization solver
        """
        # Update and plot the cycle diagram
        self.figure.suptitle(f"Optimization iteration: {iter:03d}", fontsize=14, y=0.90)
        self.plot_cycle()

        # Create a 'results' directory if it doesn't exist
        self.optimization_dir = os.path.join(self.out_dir, "optimization")
        os.makedirs(self.optimization_dir, exist_ok=True)

        # Use the solver's iteration number for the filename
        filename = os.path.join(self.optimization_dir, f"iteration_{iter:03d}.png")
        self.figure.savefig(filename)

    def _get_diagram_default_settings(self, plot_config):
        """
        Merges user-provided plot settings with default settings.

        Parameters
        ----------
        user_settings : dict
            A dictionary of user-defined plot settings.

        Returns
        -------
        dict
            A dictionary containing the merged plot settings.
        """
        default_settings = {
            "x_variable": "s",
            "y_variable": "T",
            "x_scale": "linear",
            "y_scale": "linear",
            "plot_saturation_line": True,
            "plot_critical_point": True,
            "plot_quality_isolines": False,
            "plot_pseudocritical_line": False,
            "plot_triple_point_vap": False,
            "plot_triple_point_liq": False,
            "plot_spinodal_line": False,
            # Add other default settings here
        }

        # Combine the global fluid settings with the "plots" settings
        fluid_settings = self.plot_settings.get("fluid", {})
        fluid_settings = {} if fluid_settings is None else fluid_settings
        plot_config = plot_config | fluid_settings

        # Merge with default values
        return default_settings | plot_config

    def _plot_thermodynamic_diagram(self, ax, plot_config, ax_index):
        """
        Plots or updates the thermodynamic diagram on a specified axes.

        This function sets up the axes properties according to the provided plot configuration and
        plots the phase diagram for the specified fluid. It then iterates over all the components in
        the cycle data. For heat exchangers, it plots or updates the processes for both the hot and cold sides.
        For other types of components, it plots or updates the process based on the component's data.

        The function also adjusts the plot limits if it's updating an existing plot,
        ensuring that the axes are scaled correctly to fit the new data.

        Parameters:
        ----------
        ax : matplotlib.axes.Axes
            The axes on which to plot or update the thermodynamic diagram.
        plot_config : dict
            A dictionary containing the plot settings, including variables to be plotted on the x and y axes, and scaling information.
        ax_index : int
            The index of the axes in the figure, used for identifying and updating existing plots.
        """
        # Set up axes properties
        ax.set_xlabel(LABEL_MAPPING[plot_config["x_variable"]])
        ax.set_ylabel(LABEL_MAPPING[plot_config["y_variable"]])
        ax.set_xscale(plot_config["x_scale"])
        ax.set_yscale(plot_config["y_scale"])

        # Plot phase diagram
        self.fluid.plot_phase_diagram(axes=ax, **plot_config)

        # Plot thermodynamic processes
        for name, component in self.cycle_data["components"].items():
            # Handle heat exchanger components in a special way
            if component["type"] == "heat_exchanger":
                for side in ["hot_side", "cold_side"]:
                    self._plot_cycle_process(
                        name + "_" + side, plot_config, ax, ax_index=ax_index
                    )
            else:
                self._plot_cycle_process(name, plot_config, ax, ax_index=ax_index)

        # Adjust plot limits if updating
        ax.relim(visible_only=True)
        ax.autoscale_view()

    def _plot_cycle_process(self, name, plot_settings, ax, ax_index=None):
        """
        Creates or updates the plot elements for a specific cycle process on a given axes.

        This method checks if the plot elements for the specified cycle process already exist on the given axes.
        If they exist and an axis index is provided, it updates these elements with new data. Otherwise, it creates
        new plot elements (lines and points) and stores them for future updates.

        Parameters:
        ----------
        name : str
            The name of the cycle process to plot or update.
        plot_settings : dict
            The plot settings dictionary containing settings such as x and y variables.
        ax : matplotlib.axes.Axes
            The axes on which to plot or update the cycle process.
        ax_index : int, optional
            The index of the axes in the figure, used for updating existing plots. If None, new plots are created.
        """

        # Retrieve component data
        x_data, y_data, color = self._get_process_data(
            name, plot_settings["x_variable"], plot_settings["y_variable"]
        )

        # Initialize the dictionary for this axis index if it does not exist
        if ax_index is not None:
            if ax_index not in self.graphics["process_lines"]:
                self.graphics["process_lines"][ax_index] = {}
            if ax_index not in self.graphics["state_points"]:
                self.graphics["state_points"][ax_index] = {}

        # Handle existing plot elements
        if ax_index is not None and name in self.graphics["process_lines"][ax_index]:
            if x_data is None or y_data is None:
                # Hide existing plot elements if data is None
                self.graphics["process_lines"][ax_index][name].set_visible(False)
                self.graphics["state_points"][ax_index][name].set_visible(False)
            else:
                # Update existing plot elements with new data
                self.graphics["process_lines"][ax_index][name].set_data(x_data, y_data)
                self.graphics["state_points"][ax_index][name].set_data(
                    [x_data[0], x_data[-1]], [y_data[0], y_data[-1]]
                )
                self.graphics["process_lines"][ax_index][name].set_visible(True)
                self.graphics["state_points"][ax_index][name].set_visible(True)
        elif x_data is not None and y_data is not None:
            # Create new plot elements if data is not None
            (line,) = ax.plot(
                x_data,
                y_data,
                linestyle="-",
                linewidth=1.25,
                marker="none",
                markersize=4.0,
                markeredgewidth=1.25,
                markerfacecolor="w",
                color=color,
                label=name,
                zorder=1,
            )
            (points,) = ax.plot(
                [x_data[0], x_data[-1]],
                [y_data[0], y_data[-1]],
                linestyle="none",
                linewidth=1.25,
                marker="o",
                markersize=4.0,
                markeredgewidth=1.25,
                markerfacecolor="w",
                color=color,
                zorder=2,
            )

            # Store the new plot elements
            if ax_index is not None:
                self.graphics["process_lines"][ax_index][name] = line
                self.graphics["state_points"][ax_index][name] = points

    def _get_process_data(self, name, prop_x, prop_y):
        """
        Retrieve thermodynamic data for a specified process of the cycle.

        Parameters:
        ----------
        name : str
            Name of the cycle process.
        prop_x : str
            Property name to plot on the x-axis.
        prop_y : str
            Property name to plot on the y-axis.

        Returns:
        -------
        tuple of (np.ndarray, np.ndarray, str)
            x_data: Array of data points for the x-axis.
            y_data: Array of data points for the y-axis.
            color: Color code for the plot.
        """

        # Get heat exchanger side data
        if "_hot_side" in name or "_cold_side" in name:
            # Extract the component and side from the name
            if "_hot_side" in name:
                component_name, side_1 = name.replace("_hot_side", ""), "hot_side"
                side_2 = "cold_side"
            else:
                component_name, side_1 = name.replace("_cold_side", ""), "cold_side"
                side_2 = "hot_side"

            data = self.cycle_data["components"][component_name][side_1]
            data_other_side = self.cycle_data["components"][component_name][side_2]
            is_heat_exchanger = True

        else: # Handle non-heat exchanger components           
            data = self.cycle_data["components"][name]
            is_heat_exchanger = False

        # Retrieve data
        if data["states"]["identifier"][0] == "working_fluid":
            # Components of the cycle
            x_data = data["states"][prop_x]
            y_data = data["states"][prop_y]
        elif is_heat_exchanger and prop_y == "T" and prop_x in ["h", "s"]:
            # Special case for heat exchangers
            x_data = data_other_side["states"][prop_x]
            y_data = data["states"][prop_y]
        else:
            # Other cases
            x_data = None
            y_data = None
            
        color = data["color"]

        return x_data, y_data, color

    def _plot_pinch_point_diagram(self, ax, ax_index):
        """
        Plots or updates the pinch point diagram for the thermodynamic cycle's heat exchangers.

        This method visualizes the temperature vs. heat flow rate for each heat exchanger in the cycle.
        The function is capable of plotting the diagram for the first time or updating it with the latest data
        if called subsequently. It uses a sorted approach, beginning with the heat exchanger that has the minimum
        temperature on the cold side and proceeding in ascending order of temperature.

        Parameters:
        ----------
        ax : matplotlib.axes.Axes
            The axes on which to plot or update the pinch point diagram.
        ax_index : int
            The index of the axes in the figure, used to identify and access the specific axes
            for updating the existing plot elements stored in the 'graphics' attribute.

        Notes:
        -----
        The method relies on the 'graphics' attribute of the class to store and update plot elements.
        It handles the creation of new plot elements (lines, endpoints, vertical lines) when first called
        and updates these elements with new data from 'cycle_data' during subsequent calls.
        """
        ax.set_xlabel(LABEL_MAPPING["heat"])
        ax.set_ylabel(LABEL_MAPPING["T"])

        # Initialize the graphic object dict for this axis index if it does not exist
        if ax_index not in self.graphics["pinch_point_lines"]:
            self.graphics["pinch_point_lines"][ax_index] = {}

        # Set the axes labels
        ax.set_xlabel(LABEL_MAPPING["heat"])
        ax.set_ylabel(LABEL_MAPPING["T"])

        # Extract heat exchanger names and their minimum cold side temperatures
        heat_exchangers = [
            (name, min(component["cold_side"]["states"]["T"]))
            for name, component in self.cycle_data["components"].items()
            if component["type"] == "heat_exchanger"
        ]

        # Sort heat exchangers by minimum temperature on the cold side
        sorted_heat_exchangers = sorted(heat_exchangers, key=lambda x: x[1])

        # Loop over all heat exchangers
        Q0 = 0.00
        for HX_name, _ in sorted_heat_exchangers:
            component = self.cycle_data["components"][HX_name]
            # Hot side
            c_hot = component["hot_side"]["color"]
            props_hot = component["hot_side"]["states"]
            mass_flow_hot = component["hot_side"]["mass_flow"]
            Q_hot = (props_hot["h"] - props_hot["h"][0]) * mass_flow_hot
            T_hot = props_hot["T"]

            # Cold side
            c_cold = component["cold_side"]["color"]
            props_cold = component["cold_side"]["states"]
            mass_flow_cold = component["cold_side"]["mass_flow"]
            Q_cold = (props_cold["h"] - props_cold["h"][0]) * mass_flow_cold
            T_cold = props_cold["T"]

            # Check if the plot elements for this component already exist
            if HX_name in self.graphics["pinch_point_lines"][ax_index]:
                # Update existing plot elements
                plot_elements = self.graphics["pinch_point_lines"][ax_index][HX_name]
                plot_elements["hot_line"].set_data(Q0 + Q_hot, T_hot)
                plot_elements["cold_line"].set_data(Q0 + Q_cold, T_cold)

                # Update endpoints
                plot_elements["hot_start"].set_data(Q0 + Q_hot[0], T_hot[0])
                plot_elements["hot_end"].set_data(Q0 + Q_hot[-1], T_hot[-1])
                plot_elements["cold_start"].set_data(Q0 + Q_cold[0], T_cold[0])
                plot_elements["cold_end"].set_data(Q0 + Q_cold[-1], T_cold[-1])

                # Update vertical lines
                plot_elements["start_line"].set_xdata([Q0, Q0])
                plot_elements["end_line"].set_xdata([Q0 + Q_hot[-1], Q0 + Q_hot[-1]])
            else:
                # Create new plot elements
                (hot_line,) = ax.plot(Q0 + Q_hot, T_hot, color=c_hot)
                (cold_line,) = ax.plot(Q0 + Q_cold, T_cold, color=c_cold)

                # Create endpoints
                param = {"marker": "o", "markersize": 4.0, "markerfacecolor": "white"}
                (hot_1,) = ax.plot(Q0 + Q_hot[0], T_hot[0], color=c_hot, **param)
                (hot_2,) = ax.plot(Q0 + Q_hot[-1], T_hot[-1], color=c_hot, **param)
                (cold_1,) = ax.plot(Q0 + Q_cold[0], T_cold[0], color=c_cold, **param)
                (cold_2,) = ax.plot(Q0 + Q_cold[-1], T_cold[-1], color=c_cold, **param)

                # Create vertical lines
                param = {"color": "black", "linestyle": "-", "linewidth": 0.75}
                start_line = ax.axvline(x=Q0, zorder=1, **param)
                end_line = ax.axvline(x=Q0 + Q_hot[-1], zorder=1, **param)

                # Store new plot elements
                self.graphics["pinch_point_lines"][ax_index][HX_name] = {
                    "hot_line": hot_line,
                    "cold_line": cold_line,
                    "hot_start": hot_1,
                    "hot_end": hot_2,
                    "cold_start": cold_1,
                    "cold_end": cold_2,
                    "start_line": start_line,
                    "end_line": end_line,
                }

            # Update abscissa for the next heat exchanger
            Q0 += Q_hot[-1]

        ax.set_xlim(left=0 - Q0 / 50, right=Q0 + Q0 / 50)
