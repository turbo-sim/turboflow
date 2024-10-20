%% Initialize script
% Clear the workspace
clear all
close all
clc

% Create folder to save results
results_path = 'results';
if not(isfolder(results_path))
    mkdir(results_path)
end

% Define plot settings
addpath(genpath("../common"))
set_plot_options()

% Load digitized data
df = readtable("sandia_digitize_geometry.csv");
x = df.x;
y = df.y;


%% Compute parameters from datapoints
% There is a mismatch between the area ratio that I measured from the
% figure and the area ratio that I get when computing the wedge geometry
% from the radii, the number of blades and the wedge angle when I use
% alpha_3b=71.5 degree. To be consistent with the widths measured in the
% digitized figure I adjusted alpha_3b to 78 degree

% Compute inlet/outlet radius
r_2 = 1.00;
r_3 = sqrt(x(1:4:end).^2 + y(1:4:end).^2);
r_5 = sqrt(x(3:4:end).^2 + y(3:4:end).^2);

% Compute inlet/outlet width
w_in = sqrt((x(2:4:end) - x(1:4:end)).^2 + (y(2:4:end) - y(1:4:end)).^2);
w_out = sqrt((x(4:4:end) - x(3:4:end)).^2 + (y(4:4:end) - y(3:4:end)).^2);

% Calculate bladed region inlet coordinates
x_in = (x(1:4:end) + x(2:4:end))/2;
y_in = (y(1:4:end) + y(2:4:end))/2;
r4 = sqrt(x_in.^2+y_in.^2);

% Calculate blade region outlet coordinates
x_out = (x(3:4:end) + x(4:4:end))/2;
y_out = (y(3:4:end) + y(4:4:end))/2;

% Calculate the length of the bladed region
L = sqrt((y_out - y_in).^2 + (x_out - x_in).^2);

% stagger
phi = mod((atan2(y_in, x_in) - atan2(y_out-y_in, x_out-x_in)), 2*pi);

% Caclulate the full divergence angle
div = 2*atan((w_out - w_in)./(2*L));

% Calculate the blade pitch angle (exact)
d_theta = 2*pi/17;

% Calculate the wedge angle
eps = d_theta - div;

% Print geometry
var = w_in./r_2; fprintf('%-30s:   %0.5f±%0.5f  (±%0.2f%%)\n', 'Inlet width over r_2', mean(var), std(var), std(var)/mean(var)*100)
var = w_out./r_2; fprintf('%-30s:   %0.5f±%0.5f  (±%0.2f%%)\n', 'Outlet width over r_2', mean(var), std(var), std(var)/mean(var)*100)
var = w_out./w_in; fprintf('%-30s:   %0.5f±%0.5f  (±%0.2f%%)\n', 'Area ratio', mean(var), std(var), std(var)/mean(var)*100)
var = r_3; fprintf('%-30s:   %0.5f±%0.5f  (±%0.2f%%)\n', 'Radius 3', mean(var), std(var), std(var)/mean(var)*100)
var = r4; fprintf('%-30s:   %0.5f±%0.5f  (±%0.2f%%)\n', 'Radius 4', mean(var), std(var), std(var)/mean(var)*100)
var = r_5; fprintf('%-30s:   %0.5f±%0.5f  (±%0.2f%%)\n', 'Radius 5', mean(var), std(var), std(var)/mean(var)*100)
var = L; fprintf('%-30s:   %0.5f±%0.5f  (±%0.2f%%)\n', 'Channel length', mean(var), std(var), std(var)/mean(var)*100)
var = eps*180/pi; fprintf('%-30s:   %0.5f±%0.5f  (±%0.2f%%)\n', 'Wedge angle', mean(var), std(var), std(var)/mean(var)*100)


%% Plot the datapoints
fig = figure(); hold on; box on; grid on;
axis image
for i = 1:numel(x)/4
    plot([x(1+4*(i-1)) x(2+4*(i-1)) x(3+4*(i-1)), x(4+4*(i-1)) x(1+4*(i-1))], ...
         [y(1+4*(i-1)) y(2+4*(i-1)) y(3+4*(i-1)), y(4+4*(i-1)) y(1+4*(i-1))], ...
         Color='blue', Marker='o', MarkerSize=4, MarkerFaceColor='w') 
end
plot(x_in, y_in, 'r+')
plot(x_out, y_out, 'r+')
angle = linspace(0, 2*pi, 500);
plot(mean(r_3)*cos(angle), mean(r_3)*sin(angle), 'k-')
plot(mean(r_5)*cos(angle), mean(r_5)*sin(angle), 'k-')


%% Plot the compute parameters
% Define the number of blades
i = 1:1:numel(x)/4;

% Plot the diffuser outlet radius
fig = figure(); hold on; box on;
var = r_5;
ylim([1, 2.5])
xlabel({"Blade number"; ''})
ylabel({''; "Outlet radius"})
plot(i, var, 'ko', "DisplayName", "Measured values")
plot(i, mean(var) + 0*i, 'k', 'DisplayName', "Mean value")
plot(i, mean(var)+std(var) + 0*i, 'k--', 'DisplayName', 'Typical deviation')
plot(i, mean(var)-std(var) + 0*i, 'k--', HandleVisibility='off')
legend(Location="best")

% Plot the divergence wedge angle
fig = figure(); hold on; box on;
var = div*180/pi;
xlabel({"Blade number"; ''})
ylabel({''; "Divergence full angle"})
ylim([0, 30])
plot(i, var, 'ko', "DisplayName", "Measured values")
plot(i, mean(var) + 0*i, 'k', 'DisplayName', "Mean value")
plot(i, mean(var)+std(var) + 0*i, 'k--', 'DisplayName', 'Typical deviation')
plot(i, mean(var)-std(var) + 0*i, 'k--', HandleVisibility='off')
legend(Location="best")

% Plot the diffuser wedge angle
fig = figure(); hold on; box on;
var = eps*180/pi;
xlabel({"Blade number"; ''})
ylabel({''; "Wedge full angle"})
ylim([0, 30])
plot(i, var, 'ko', "DisplayName", "Measured values")
plot(i, mean(var) + 0*i, 'k', 'DisplayName', "Mean value")
plot(i, mean(var)+std(var) + 0*i, 'k--', 'DisplayName', 'Typical deviation')
plot(i, mean(var)-std(var) + 0*i, 'k--', HandleVisibility='off')
legend(Location="best")

% Plot the diffuser stagger angle
fig = figure(); hold on; box on;
var = phi*180/pi;
xlabel({"Blade number"; ''})
ylabel({''; "Stagger angle"})
% ylim([0, 30])
plot(i, var, 'ko', "DisplayName", "Measured values")
plot(i, mean(var) + 0*i, 'k', 'DisplayName', "Mean value")
plot(i, mean(var)+std(var) + 0*i, 'k--', 'DisplayName', 'Typical deviation')
plot(i, mean(var)-std(var) + 0*i, 'k--', HandleVisibility='off')
legend(Location="best")

% Plot the area ratio
fig = figure(); hold on; box on;
var = w_out./w_in;
xlabel({"Blade number"; ''})
ylabel({''; "Area ratio"})
% ylim([0, 30])
plot(i, var, 'ko', "DisplayName", "Measured values")
plot(i, mean(var) + 0*i, 'k', 'DisplayName', "Mean value")
plot(i, mean(var)+std(var) + 0*i, 'k--', 'DisplayName', 'Typical deviation')
plot(i, mean(var)-std(var) + 0*i, 'k--', HandleVisibility='off')
legend(Location="best")

