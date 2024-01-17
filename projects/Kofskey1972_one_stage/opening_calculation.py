# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:10:38 2024

@author: lasseba
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

# Read the Excel file into a DataFrame
file_path = 'Kofskey_1972_geometry.xlsx'
sheet_names = ['Stator coordinates', 'Rotor coordinates']
df = pd.read_excel(file_path, sheet_name = sheet_names)
pitch = {"s1" : 1.829370629,
         "s2" : 1.484353741,
         "r1" : 1.523976608,
         "r2" : 1.45060241}

xi = {"s1" : 90-43.03,
         "s2" : 90-31.05,
         "r1" : 90-28.72,
         "r2" : 90-21.75}

airfoil = 'r2'
plot_original = True # Plot airfoil in coordinate system along chord

# Get pitch and stagger angle value
pitch = pitch[airfoil]
xi = xi[airfoil]*np.pi/180

# Get sheet that corresponds to the airfoil
if airfoil.startswith('s'):
    df = df[sheet_names[0]]
    
elif airfoil.startswith('r'):
    df = df[sheet_names[1]]

# Get the airfoil specific data
df = df[["x", f"{airfoil}_YU", f"{airfoil}_YL"]]
df = df[df[f"{airfoil}_YU"]>0] # Avoid NANs

# Sort the DataFrame based on the 'x' column
df = df.sort_values(by='x')

# Extract the x, y_upper, and y_lower coordinates 
x_values = df['x'].values
y_upper_values = df[f'{airfoil}_YU'].values
y_lower_values = df[f'{airfoil}_YL'].values

# Perform cubic spline interpolation
cs_upper = CubicSpline(x_values, y_upper_values, bc_type='not-a-knot')
cs_lower = CubicSpline(x_values, y_lower_values, bc_type='not-a-knot')

# Generate a higher-resolution x array for smoother interpolation
x_interp = np.linspace(x_values.min(), x_values.max(), 1000)

# Interpolate y values for the upper and lower surfaces
y_upper_interp = cs_upper(x_interp)
y_lower_interp = cs_lower(x_interp)

if plot_original:
    fig, ax = plt.subplots()
    ax.plot(x_interp, y_upper_interp, color='green')
    ax.plot(x_interp, y_lower_interp, color='red')

# Shift coordinate system
def shift_coordinate(x_values, y_lower_values, y_upper_values, xi):
    x_lower = x_values*np.cos(xi) - y_lower_values*np.sin(xi)
    x_upper = x_values*np.cos(xi) - y_upper_values*np.sin(xi)
    y_lower_values = x_values*np.sin(xi) + y_lower_values*np.cos(xi)
    y_upper_values = x_values*np.sin(xi) + y_upper_values*np.cos(xi)

    return x_lower, x_upper, y_lower_values, y_upper_values

x_lower, x_upper, y_lower_values, y_upper_values = shift_coordinate(x_values, y_lower_values, y_upper_values, xi)
x_interp_lower, x_interp_upper, y_lower_interp, y_upper_interp = shift_coordinate(x_interp, y_lower_interp, y_upper_interp, xi)

N = len(x_interp_lower)
distance = np.zeros((N,N))

for i in range(N):
    for j in range(N):
        x1 = x_interp_lower[i]
        y1 = y_lower_interp[i]
        x2 = x_interp_upper[j] + pitch
        y2 = y_upper_interp[j]
        
        distance[i,j] = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
opening = np.min(distance)
index = np.where(distance == opening)
i = index[0][0]
j = index[1][0]

# Plot airfoil
fig1, ax1 = plt.subplots()

# Scatter original data points
ax1.scatter(x_upper, y_upper_values, label='Original Upper Surface', color='blue')
ax1.scatter(x_lower, y_lower_values, label='Original Lower Surface', color='orange')

# Plot interpolated airfoil
ax1.plot(x_interp_upper, y_upper_interp, label='Interpolated Upper Surface', color='green')
ax1.plot(x_interp_lower, y_lower_interp, label='Interpolated Lower Surface', color='red')

# Plot adjecent airfoil
ax1.plot(x_interp_upper + pitch, y_upper_interp, color='green')
ax1.plot(x_interp_lower + pitch, y_lower_interp, color='red')

# Plot opening
ax1.plot([x_interp_lower[i], x_interp_upper[j] + pitch], [y_lower_interp[i], y_upper_interp[j]], label = 'opening')

# Add labels and legend
ax1.set_xlabel('x-coordinate')
ax1.set_ylabel('y-coordinate')
ax1.set_title('Airfoil Interpolation')
ax1.legend()

# Show the plot
plt.show()

    

