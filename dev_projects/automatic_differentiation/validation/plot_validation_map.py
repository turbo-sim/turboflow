import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import turboflow as tf


tf.set_plot_options()

df = pd.read_excel("./output_map/performance_latest.xlsx", sheet_name= "overall")
df_exp = pd.read_excel("./validation_cases_summary.xlsx")

markerlist = ["o", "+", "*", "s"]
# fig, ax = plt.subplots()
# for omega in df["angular_speed"][df["angular_speed"] > 0.6*1627]:
#     ax.plot(df["PR_ts"][df["angular_speed"]==omega], df["efficiency_ts"][df["angular_speed"]==omega])
#     ax.plot(df_exp["pressure_ratio_ts"][np.isclose(df_exp["omega"], omega)], df_exp["efficiency_ts"][np.isclose(df_exp["omega"], omega)], linestyle= "None", marker="o")
#     # ax.set_ylim([2.55, 2.85])

# fig, ax = plt.subplots()
# for omega in df["angular_speed"][df["angular_speed"] > 0.6*1627]:
#     ax.plot(df["PR_ts"][df["angular_speed"]==omega], df["torque"][df["angular_speed"]==omega])
#     ax.plot(df_exp["pressure_ratio_ts"][np.isclose(df_exp["omega"], omega)], df_exp["torque"][np.isclose(df_exp["omega"], omega)], linestyle= "None", marker="o")
#     # ax.set_ylim([2.55, 2.85])

# fig, ax = plt.subplots()
# for omega in df["angular_speed"][df["angular_speed"] > 0.6*1627]:
#     ax.plot(df["PR_ts"][df["angular_speed"]==omega], df["mass_flow_rate"][df["angular_speed"]==omega])
#     ax.plot(df_exp["pressure_ratio_ts"][np.isclose(df_exp["omega"], omega)], df_exp["mass_flow_rate"][np.isclose(df_exp["omega"], omega)], linestyle= "None", marker="o")
#     # ax.set_ylim([2.55, 2.85])

# Create a figure and axis
fig, ax = plt.subplots()

# Define a color palette with shades of red, orange, and brown
# You can manually define these colors or generate them using matplotlib's color options
# colors = ['#D7301F', '#FC8D59', '#FEE08B', '#D8B365']  # Red, Orange, Yellow, Dark Brown
colors = plt.get_cmap('plasma_r')(np.linspace(0.2, 1.0, 4))
num_omegas = len(colors)

# Lists to store the handles and labels for the legend
handles_sim = []  # Handles for simulation lines
labels_sim = []   # Labels for simulation lines
handles_exp = []  # Handles for experimental points
labels_exp = []   # Labels for experimental points

# Assuming you already have `df` and `df_exp` for the data
unique_omegas = df["angular_speed"][df["angular_speed"] > 0.6 * 1627].unique()

# Loop through omega values
for idx, omega in enumerate(unique_omegas):
    
    # Filter data for the current omega
    sim_x = df["PR_ts"][df["angular_speed"] == omega]
    sim_y = df["efficiency_ts"][df["angular_speed"] == omega]
    
    # Plot the simulation line with the chosen color
    line, = ax.plot(sim_x, sim_y, color=colors[idx], label=f'{int(np.round((omega/1627)*100))}')
    
    # Add the line handle and label to the simulation legend list
    handles_sim.append(line)
    labels_sim.append(f'{int(np.round((omega/1627)*100))}')
    
    # Experimental data for the current omega
    exp_x = df_exp["pressure_ratio_ts"][np.isclose(df_exp["omega"], omega)]
    exp_y = df_exp["efficiency_ts"][np.isclose(df_exp["omega"], omega)]
    
    # Define markers for experimental points (circle, square, triangle, diamond)
    markers = ['o', 's', '^', 'D']  
    exp_points, = ax.plot(exp_x, exp_y, linestyle="None", marker=markers[idx % len(markers)], color=colors[idx], label=f'{int(np.round((omega/1627)*100))}')
    
    # Add the experimental points handle and label to the experimental legend list
    handles_exp.append(exp_points)
    labels_exp.append(f'{int(np.round((omega/1627)*100))}')

# Set the labels for axes
ax.set_xlabel(r'Total-to-static pressure ratio [$p_{0,\text{in}} / p_{\text{out}}$]', fontsize=22)
ax.set_ylabel(r'Total-to-static efficiency($\%$)', fontsize=22)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

# Combine the handles and labels for simulation and experimental data
handles = handles_sim + handles_exp
labels = labels_sim + labels_exp

# Add the legend to the plot with the requested location (lower right) and title
# Set the number of columns to 2 for the legend: one column for simulation lines, another for experimental points
legend = ax.legend(handles=handles, labels=labels, title="Percent of design\nangular speed", loc="lower left", fontsize=10, ncol=2)

# Center the title of the legend manually
legend.get_title().set_horizontalalignment('center')

# Optionally, you can adjust the y-axis limits
ax.set_ylim([40, 90])
ax.set_xlim([1.4, 4.7])

# Adjust the layout to make room for the legend
plt.tight_layout()

# Show the plot
plt.show()
