# Flow variables at the throat
density = 1
dynamic_viscosity = 1.81e-5
velocity = 200

# Distance from the leading edge to the throat
opening = 3e-3
chord = 10e-3
x = 0.9*chord

# Reynolds number based on distance from leading edge to throat
Re = density*velocity*x/dynamic_viscosity

# Boundary layer thickness at the throat based on flat-plate theory
delta = 0.16/(Re **(1/7)) * x

# Displacement thickness at the throat
delta_star_1 = delta/8
delta_star_2 = 0.020/(Re **(1/7)) * x
delta_star_3 = 0.048/(Re **(1/5)) * x

# Print estimation
print(f"{'Boundary layer thickness at the throat:':40s} {delta*1000:0.4f} mm")
print(f"{'Displacement thickness at the throat:':40s} {delta_star_1*1000:0.4f} mm")
print(f"{'Displacement thickness at the throat:':40s} {delta_star_2*1000:0.4f} mm")
print(f"{'Displacement thickness at the throat:':40s} {delta_star_3*1000:0.4f} mm")
# print(f"{'Fraction of throat opening:':40s} {delta_star/opening*100:0.2f} %")

