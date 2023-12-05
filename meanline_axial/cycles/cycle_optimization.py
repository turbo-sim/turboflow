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

from . import cycle_models as cycles


utilities.set_plot_options()
COLORS_MATLAB = utilities.COLORS_MATLAB
LABEL_MAPPING = {
    "s": "Entropy [J/kg/K]",
    "T": "Temperature [K]",
    "p": "Pressure [Pa]",
    "h": "Enthalpy [J/kg]",
    "d": "Density [kg/m$^3$]",
    "a": "Speed of sound [m/s]",
    "Z": "Compressibility factor [-]",
}

class BraytonCycleProblem(solver.OptimizationProblem):
    """
    A class to represent a Brayton Cycle optimization problem.

    This class provides functionalities to load and update the configuration for
    the Brayton Cycle, and to perform interactive plotting based on the current
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
        Constructs all the necessary attributes for the BraytonCycleProblem object.

        Parameters
        ----------
        config : dict
            Dictionary containing the configuration for the Brayton Cycle problem.
        """
        # Initialize variables
        self.figure = None
        self.figure_TQ = None
        self.configuration = copy.deepcopy(configuration)
        self.update_configuration(self.configuration)

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
            Dictionary containing the new configuration for the Brayton Cycle problem.
        """
        conf = configuration
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
        self.fixed_parameters["cycle_fluid"] = {
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
        config = utilities.read_configuration_file(config_file, validate=False)
        self.update_configuration(config["problem_formulation"])
        self._calculate_special_points()

    def get_optimization_values(self, x):
        # Create dictionary from array of normalized design variables
        self.variables = dict(zip(self.keys, x))
        vars_physical = self._scale_normalized_to_physical(self.variables)

        # Evaluate model
        self.cycle_data = cycles.evaluate_recuperated_cycle(
            vars_physical,
            self.fixed_parameters,
            self.constraints,
            self.objective_function,
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
                    lower = utilities.render_and_evaluate(lower_expr, self.fixed_parameters)
                else:
                    lower = lower_expr
                if isinstance(upper_expr, str):
                    upper = utilities.render_and_evaluate(upper_expr, self.fixed_parameters)
                else:
                    upper = upper_expr

                vars_physical[key] = scale_value(lower, upper, value)

            except (KeyError, ValueError) as e:
                raise ValueError(f"Error processing bounds for '{key}': {e}")

        return vars_physical


    def plot_cycle_diagram(self):
        """
        Plots or updates the thermodynamic cycle diagram based on current settings.

        This function is designed to create a new cycle diagram or update an existing one. It is
        especially useful in scenarios where the cycle diagram needs to be dynamically refreshed.
        For instance, during optimization steps, this function can be used as a callback to
        continually update the plot with new optimization data. It is also utilized by the
        `plot_cycle_diagram_realtime` method to refresh the plot based on the latest values
        from the YAML configuration file.

        The function checks if a cycle diagram already exists. If it does, the diagram is updated
        with new data; otherwise, a new diagram is created. This approach ensures that the plot
        reflects the most current state of the cycle, including any changes made during
        optimization or adjustments to the configuration settings.

        The plotting settings are read from `self.plot_settings`, which are updated with
        default values for any unspecified settings.
        """

        # Use default values for unspecified settings
        self.plot_settings = self._get_plot_settings(self.plot_settings)

        # Create new figure or update data of existing figure
        if not (self.figure and plt.fignum_exists(self.figure.number)):
            # Initialize figure
            self.figure, self.ax = plt.subplots(figsize=(6.0, 4.8))
            self.ax.set_xlabel(LABEL_MAPPING[self.plot_settings["x_variable"]])
            self.ax.set_ylabel(LABEL_MAPPING[self.plot_settings["y_variable"]])
            self.ax.set_xscale(self.plot_settings["x_scale"])
            self.ax.set_yscale(self.plot_settings["y_scale"])
            self.process_lines = {}
            self.state_points = {}

            # Plot phase diagram
            self.fluid.plot_phase_diagram(axes=self.ax, **self.plot_settings)

            # Loop over all cycle processes
            for name in self.cycle_data["processes"]:
                self._plot_cycle_component(name)

        else:  # The figure already exists
            # Update axes
            self.ax.set_xlabel(LABEL_MAPPING[self.plot_settings["x_variable"]])
            self.ax.set_ylabel(LABEL_MAPPING[self.plot_settings["y_variable"]])
            self.ax.set_xscale(self.plot_settings["x_scale"])
            self.ax.set_yscale(self.plot_settings["y_scale"])

            # Plot phase diagram
            self.fluid.plot_phase_diagram(axes=self.ax, **self.plot_settings)

            # Update all cycle processes
            for name in self.cycle_data["processes"]:
                self._update_cycle_component(name)

            # Adjust the plot limits
            self.ax.relim()
            self.ax.autoscale_view()

        # Give some time to update plot
        self.figure.tight_layout(pad=1)
        plt.draw()
        plt.pause(0.05)

    def plot_optimization_history(self, x, iter):
        """
        Plot and save the thermodynamic cycle at different optimization iterations

        This function is intended to be passed as callback argument to the optimization solver
        """
        # Update and plot the cycle diagram
        self.ax.set_title(f"Optimization iteration: {iter:03d}")
        self.plot_cycle_diagram()

        # Create a 'results' directory if it doesn't exist
        self.optimization_dir = os.path.join(self.out_dir, "optimization")
        os.makedirs(self.optimization_dir, exist_ok=True)

        # Use the solver's iteration number for the filename
        filename = os.path.join(self.optimization_dir, f"iteration_{iter:03d}.png")
        self.figure.savefig(filename)

    def plot_cycle_diagram_realtime(self, configuration_file, update_interval=0.1):
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
        print(" Creating Brayton cycle interactive plot")
        print("-" * 80)
        print(f" The current configuration file is: '{configuration_file}'")
        print(" Modify the configuration file and save it to update the plot")
        print(" Try to find a good initial guess for the optimization.")
        print(" Using a feasible initial guess improves optimization convergence.")
        print(" Press 'enter' or close the figure to continue.")
        print("-" * 80)

        # Update the plot until termination signal
        while not self.enter_pressed:
            # Read the configuration file
            self.load_configuration_file(configuration_file)
            self.get_optimization_values(self.x0)
            self.plot_cycle_diagram()

            # Wait for the specified interval before updating again
            time.sleep(update_interval)

            # Exit interactive plotting when the user closes the figure
            if not plt.fignum_exists(self.figure.number):
                break

            # Exit interactive plotting when the user presses enter
            if self.enter_pressed:
                break

    def _get_component_data(self, name, prop_x, prop_y):
        """
        Retrieve thermodynamic data for a specified component of the cycle.

        The x-property of the heating and cooling fluids is modified to match the working fluid values
        This looks good for plots with temperature in the y-axis and entropy/enthalpy in the x-axis

        Parameters
        ----------
        name : str
            Name of the cycle component.
        prop_x : str
            Property name to plot on the x-axis.
        prop_y : str
            Property name to plot on the y-axis.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray, str)
            x_data: Array of data points for the x-axis.
            y_data: Array of data points for the y-axis.
            color: Color code for the plot.
        """
        data = self.cycle_data["processes"]
        if name == "heater_hot_side":
            x_data = data["heater_cold_side"]["states"][prop_x]
            y_data = data[name]["states"][prop_y]
            color = COLORS_MATLAB[6]
        elif name == "cooler_cool_side":
            x_data = data["cooler_hot_side"]["states"][prop_x]
            y_data = data[name]["states"][prop_y]
            color = COLORS_MATLAB[0]
        else:
            x_data = data[name]["states"][prop_x]
            y_data = data[name]["states"][prop_y]
            color = COLORS_MATLAB[1]

        return x_data, y_data, color

    def _plot_cycle_component(self, name, linewidth=1.00):
        """
        Plots a the states of a cycle component on the current axes.

        Parameters
        ----------
        name : str
            Name of the cycle component to plot.
        linewidth : float, optional
            Line width for the plot, by default 1.00.
        """
        x_data, y_data, color = self._get_component_data(
            name,
            self.plot_settings["x_variable"],
            self.plot_settings["y_variable"],
        )
        x_points = [x_data[0], x_data[-1]]
        y_points = [y_data[0], y_data[-1]]

        (self.process_lines[name],) = self.ax.plot(
            x_data,
            y_data,
            linestyle="-",
            linewidth=linewidth,
            marker="none",
            markersize=4.5,
            markeredgewidth=linewidth,
            markerfacecolor="w",
            color=color,
            label=name,
        )

        (self.state_points[name],) = self.ax.plot(
            x_points,
            y_points,
            linestyle="none",
            linewidth=linewidth,
            marker="o",
            markersize=4.5,
            markeredgewidth=linewidth,
            markerfacecolor="w",
            color=color,
        )

    def _update_cycle_component(self, name):
        """
        Updates the plot data for an existing cycle component.

        Parameters
        ----------
        name : str
            Name of the cycle component to update.
        """
        x_data, y_data, _ = self._get_component_data(
            name,
            self.plot_settings["x_variable"],
            self.plot_settings["y_variable"],
        )
        x_points = [x_data[0], x_data[-1]]
        y_points = [y_data[0], y_data[-1]]

        self.process_lines[name].set_data(x_data, y_data)
        self.state_points[name].set_data(x_points, y_points)

    @staticmethod
    def _get_plot_settings(user_settings):
        """
        Merges user-provided plot settings with default settings.

        Any setting not specified by the user will be set to a default value.

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
        return {**default_settings, **user_settings}

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
        headers = ["Process"]
        units_row = ["Units"]

        for key in variable_map:
            headers.append(variable_map[key]["name"])
            units_row.append(variable_map[key]["unit"])

        # Iterate over each process in the dictionary
        for process_name, process in self.cycle_data["processes"].items():
            state_in = process["state_in"].to_dict()
            state_out = process["state_out"].to_dict()

            # Append the data for state_in and state_out to the rows list
            data_rows.append(
                [process_name + "_in"]
                + [state_in.get(key, None) for key in variable_map]
            )
            data_rows.append(
                [process_name + "_out"]
                + [state_out.get(key, None) for key in variable_map]
            )

        # Create a DataFrame with data rows
        df = pd.DataFrame(data_rows, columns=headers)

        # Insert the units row
        df.loc[-1] = units_row  # Adding a row
        df.index = df.index + 1  # Shifting index
        df = df.sort_index()  # Sorting by index

        # Export to Excel
        df.to_excel(os.path.join(self.out_dir, filename), index=False, header=True)
