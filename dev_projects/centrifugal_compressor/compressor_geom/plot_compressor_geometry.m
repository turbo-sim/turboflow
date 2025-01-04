%% Initialize script
clear all
close all
clc

% Define plot settings
set_plot_options()
save_figures = false;
results_path = 'results';
if not(isfolder(results_path))
    mkdir(results_path)
end


% Define case name
addpath(genpath("./cases"))
addpath(genpath("./functions"))
case_name = 'sandia_compressor';
% case_name = 'eckardt_impeller_A';
% case_name = 'eckardt_impeller_O';
% case_name = 'NASA_vaned';

% Load case geometry
pg = eval(case_name);
options = options_compr();
geom = complete_geometry_compr(pg, options);


%% Axial-radial view
% Add backplate realistic geometry
fig = figure(); hold on; box on; grid on;
axis image;
ax = gca;
ax.Layer = 'top';
ax.GridLineStyle = "-";
ax.GridColor = 'k';
ax.GridAlpha = 0.10;
xlabel({''; 'Axial coordinate (m)'})
ylabel({'Radial coordinate (m)'; ''})
xtickformat('%0.3f')
ytickformat('%0.3f')
plot_compressor_meridional_plane(geom, axial_offset=0.0, edge_width=0.5, color_hub=0.90*[1,1,1], color_blades=0.75*[1,1,1], symmetric=false)
if save_figures
    exportgraphics(fig, fullfile(results_path, [case_name, '_axial_radial.png']), 'Resolution', 500);
end


%% Blade-to-blade view
fig = figure(); hold on; box on; grid on;
axis image;
ax = gca;
ax.Layer = 'top';
ax.GridLineStyle = "-";
ax.GridColor = 'k';
ax.GridAlpha = 0.10;
title({"Hub section blade-to-blade plane"; ''}, FontSize=11)
xlabel({''; '$\hat{m} = \int{\frac{\mathrm{d}m}{r}}$ coordinate'})
ylabel({'$\theta$ coordinate'; ''})
xtickformat('%0.2f')
ylim([0, pi])
plot_compressor_blade_to_blade(geom, section="hub", color_blades=0.75*[1,1,1], edge_width=0.5)
if save_figures
    exportgraphics(fig, fullfile(results_path, [case_name, '_blade_to_blade.png']), 'Resolution', 500);
end

fig = figure(); hold on; box on; grid on;
axis image;
ax = gca;
ax.Layer = 'top';
ax.GridLineStyle = "-";
ax.GridColor = 'k';
ax.GridAlpha = 0.10;
title({"Shroud section blade-to-blade plane"; ''}, FontSize=11)
xlabel({''; '$\hat{m} = \int{\frac{\mathrm{d}m}{r}}$ coordinate'})
ylabel({'$\theta$ coordinate'; ''})
xtickformat('%0.2f')
ylim([0, pi])
plot_compressor_blade_to_blade(geom, section="shroud", color_blades=0.75*[1,1,1], edge_width=0.5)
if save_figures
    exportgraphics(fig, fullfile(results_path, [case_name, '_blade_to_blade.png']), 'Resolution', 500);
end


%% Tangential-radial view
fig = figure(); hold on; box on; grid on;
axis image;
ax = gca;
ax.GridLineStyle = "-";
ax.GridColor = 'k';
ax.GridAlpha = 0.10;
xlabel({'$z$ coordinate (m)'})
ylabel({'$x$ coordinate (m)'})
zlabel({'$y$ coordinate (m)'})
ax.YDir = "reverse";
xtickformat('%0.3f')
ytickformat('%0.3f')
ztickformat('%0.3f')
view(-90, 0)
plot_compressor_impeller_3D(geom, color_hub=0.90*[1,1,1], color_blades=0.75*[1,1,1], edge_width=0.5)
plot_compressor_diffuser_3D(geom, color_hub=0.90*[1,1,1], color_blades=0.75*[1,1,1], edge_width=0.5, plot_diffuser_channels=false)
if save_figures
    exportgraphics(fig, fullfile(results_path, [case_name, '_tangential_radial.png']), 'Resolution', 500);
end


%% Plot blade in 3D
% Initialize figure
fig = figure(); hold on; box on; grid on;
axis image;
ax = gca;
% axis off
xlabel({'$z$ coordinate (m)'})
ylabel({'$x$ coordinate (m)'})
zlabel({'$y$ coordinate (m)'})
ax.YDir = "reverse";
view(-70, 10)
lightangle(-100,+30)
lightangle(100,+30)
plot_compressor_impeller_3D(geom, color_hub=0.95*[1, 1, 1], color_blades=0.80*[1, 1, 1], edge_width=0.5)
plot_compressor_diffuser_3D(geom, color_hub=0.95*[1, 1, 1], color_blades=0.80*[1, 1, 1], edge_width=0.5, plot_diffuser_channels=false)
z_min = -max(2*geom.L_inducer, geom.L_z);
z_max = +1.25*(geom.L_z+geom.L_shaft);
a = 1.2*geom.r_6;
axis([z_min, z_max, -a, a, -a, a])
if save_figures
    exportgraphics(fig, fullfile(results_path, [case_name, '_3D.png']), 'Resolution', 500);
end


