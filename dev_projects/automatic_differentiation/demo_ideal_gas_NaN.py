import turboflow as tf

# Constants
R = 287.0  # Specific gas constant for air (J/(kg*K))
gamma = 1.41  # Specific heat ratio for air
T_ref = 288.15  # Reference temperature (K)
P_ref = 101306.33  # Reference pressure (Pa)
s_ref = 1659.28  # Reference entropy (J/(kg*K))
myu_ref = 1.789e-5  # Reference dynamic viscosity (Kg/(m*s))
S_myu = 110.56  # Sutherland's constant for viscosity (K)
k_ref = 0.0241  # Reference thermal conductivity (W/(m*K))
S_k = 194  # Sutherland's constant for thermal conductivity (K)
h_ref = gamma*R / (gamma - 1) * T_ref


# Calculations are okay for positive absolute temperatures
s = s_ref
T = T_ref-250
cp = gamma*R/(gamma-1)
h = cp*(T-T_ref) + h_ref
# Calculate thermodynamic properties
properties = tf.perfect_gas_props("HmassSmass_INPUTS", h, s)
tf.print_dict(properties)

# Computations fail for negative absolute temperature
s = s_ref
T = T_ref-300
cp = gamma*R/(gamma-1)
h = cp*(T-T_ref) + h_ref
# Calculate thermodynamic properties
properties = tf.perfect_gas_props("HmassSmass_INPUTS", h, s)
tf.print_dict(properties)


gamma = 1.4
T_in = 300
p_in = 101325
p_out = p_in / 100
T_out_s = T_in * (p_out / p_in) ** ((gamma-1)/ gamma)
print("T_out_s", T_out_s)