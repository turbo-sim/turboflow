# import CoolProp as CP
import CoolProp.CoolProp as CP

# Define the fluid
fluid = 'Water'

# Define the state of the fluid
temperature = 373.15  # Temperature in Kelvin (100 °C)
pressure = CP.PropsSI('P', 'T', temperature, 'Q', 0, fluid)  # Saturation pressure

# Calculate properties using low-level interface
# Create an instance of the fluid
fluid_instance = CP.AbstractState('HEOS', fluid)

# Set the state of the fluid
fluid_instance.update(CP.HA, 0, pressure)  # Using enthalpy and pressure

# Retrieve various properties
enthalpy = fluid_instance.hmass()  # Specific enthalpy in J/kg
entropy = fluid_instance.smass()    # Specific entropy in J/kg·K
density = fluid_instance.rhomass()   # Density in kg/m³

# Print the results
print(f"Saturation Pressure at {temperature} K: {pressure} Pa")
print(f"Specific Enthalpy: {enthalpy} J/kg")
print(f"Specific Entropy: {entropy} J/kg·K")
print(f"Density: {density} kg/m³")
