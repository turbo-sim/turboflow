
import numpy as np
import turboflow as tf
import matplotlib.pyplot as plt

vaned_diffuser = {
    "radius_out" : 183.76/2*1e-3,
    "radius_in" : 150.22/2*1e-3,
    "width_in" : 6.29e-3,
    "width_out" : 6.29e-3,
    "leading_edge_angle" : 77.73, # From NASA compressors
    "trailing_edge_angle" : 34.0, # From NASA compressors
    "number_of_vanes" : 21,
}


# # Load geometry
# r_out = vaned_diffuser["radius_out"] # Radius out
# r_in = vaned_diffuser["radius_in"] # Radius in
# b_in = vaned_diffuser["width_in"] # Channel width in
# b_out = vaned_diffuser["width_out"] # Channel width out
# theta_in = vaned_diffuser["leading_edge_angle"] # Leading edge blade angle
# theta_out = vaned_diffuser["trailing_edge_angle"] # Trailing edge blade angle
# z = vaned_diffuser["number_of_vanes"] # Number of vanes

# # Calculate geometry
# theta_avg = (theta_in + theta_out)/2 # Mean blade angle
# camber = abs(theta_in - theta_out) # Camber angle
# solidity = z*(r_out -r_in)/(2*np.pi*r_in*tf.cosd(theta_avg)) # Solidity 
# loc_camber_max = (2-abs(theta_in - theta_avg)/camber)/3 # Location of max camber

# # Guess inlet flow angle
# N = 100
# alpha_in = np.linspace(75.00, 80.00, N)

# # Calculate incidence
# alpha_inc = alpha_in - theta_in

# # Calculate deviation angle and absolute flow angle
# delta_0 = camber*(0.92*loc_camber_max**2 + 0.02*theta_out)/(np.sqrt(solidity) - 0.02*camber)
# d_delta = np.exp(((1.5-(90-theta_in)/60)**2-3.3)*solidity)
# alpha_out = theta_out + delta_0 + d_delta*alpha_inc

# Load geometry
r_out = vaned_diffuser["radius_out"] # Radius out
r_in = vaned_diffuser["radius_in"] # Radius in
b_in = vaned_diffuser["width_in"] # Channel width in
b_out = vaned_diffuser["width_out"] # Channel width out
theta_in = 90 - vaned_diffuser["leading_edge_angle"] # Leading edge blade angle
theta_out = 90 - vaned_diffuser["trailing_edge_angle"] # Trailing edge blade angle
z = vaned_diffuser["number_of_vanes"] # Number of vanes

theta_in = 10
theta_out = 20

# Calculate geometry
theta_avg = (theta_in + theta_out)/2 # Mean blade angle
camber = (theta_out-theta_in) # Camber angle
solidity = z*(r_out -r_in)/(2*np.pi*r_in*tf.sind(theta_avg)) # Solidity 
loc_camber_max = (2-(theta_avg - theta_in)/camber)/3 # Location of max camber

# Guess inlet flow angle
N = 1000
alpha_in = np.linspace(00.00, 20.00, N)

# Calculate incidence
alpha_inc =  theta_in - alpha_in

# Calculate deviation angle and absolute flow angle
delta_0 = camber*(0.92*loc_camber_max**2 + 0.02*(90-theta_out))/(np.sqrt(solidity) - 0.02*camber)
d_delta = np.exp(((1.5-theta_in/60)**2-3.3)*solidity)
alpha_out = theta_out - delta_0 - d_delta*alpha_inc
print((0.92*loc_camber_max**2 + 0.02*(90-theta_out)))
print(np.sqrt(solidity) - 0.02*camber)
print(solidity)

fig, ax = plt.subplots()
ax.plot(alpha_inc, delta_0 + d_delta*alpha_inc)
plt.show()