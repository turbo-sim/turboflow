function solution = vaneless_diffuser_model(parameters)

% This function contains the diffuser flow model

% If diffuser_model = '1D_flow'
% Solve the ODE system describing the flow in a vaneless annular channel
% 1) steady
% 2) compressible
% 3) axisymmetric flow
% 4) with heat transfer and friction
%

%% Input parameters
% Design parameters
Cf = parameters.Cf;          % Skin friction coefficient

% Geometric parameters
phi = parameters.phi;        % Mean wall cant angle
div = parameters.div;        % Wall divergence semi-angle
r_in = parameters.r_in;      % Radius at the inlet
b_in = parameters.b_in;      % Channel width at the inlet
r_out = parameters.r_out;    % Radius at the outlet

% Inlet state
fluid = parameters.fluid;
p0_in = parameters.p0_in;
T0_in = parameters.T0_in;
Ma_in = parameters.Ma_in;
s_in = PropsSI_scalar('S', 'P', p0_in, 'T', T0_in, fluid);
p_in = static_pressure_from_mach(p0_in, T0_in, Ma_in, fluid);
a_in = PropsSI_scalar('A', 'P', p_in, 'S', s_in, fluid);
d_in = PropsSI_scalar('D', 'P', p_in, 'S', s_in, fluid);

% Inlet velocity
alpha_in = parameters.alpha_in;
v_in = Ma_in * a_in;
v_m_in = v_in*cos(alpha_in);
v_t_in = v_in*sin(alpha_in);


%% Compute the geometry of the diffuser
% Define the end of the intergration interval (radius increment)
m_out = r_out - r_in;

% Define the initial conditions
x_in = 0;
theta_in = 0;  % Streamline initial coordinate
U0 = [v_m_in v_t_in d_in p_in s_in, theta_in];

% Use the options RelTol and AbsTol to set the integration tolerance
options = odeset('RelTol',1e-4, 'AbsTol',1e-4);

% Integrate the ode system using ode45
% Use m and U and variables and the rest of inputs as extra parameters
[m,U] = ode45(@(m,U)ode_diffuser(m,U,phi,div,r_in,b_in,x_in,Cf,fluid),[0,m_out],U0,options);

% Rename the solution to the other calculations
v_m = U(:,1);
v_t = U(:,2);
d = U(:,3);
p = U(:,4);
s_gen = U(:,5);
theta = U(:,6);

% Compute the entropy and stagnation enthalpy using the equations of state
s = PropsSI_array('S', 'P', p, 'D', d, fluid);
h = PropsSI_array('H', 'P' ,p, 'D', d, fluid);
v = sqrt(v_m.^2 + v_t.^2);
h0 = h + v.^2/2;

% Compute Mach number
a = PropsSI_array('A', 'P', p, 'D', d, fluid);
Ma_t = v_t./a;
Ma_m = v_m./a;
Ma = v./a;

% Compute the geometry of the diffuser
r = r_fun(r_in,phi,m);
x = x_fun(x_in,phi,m);
b = b_fun(b_in,div,m);
A = 2*pi*r.*b;
RR = r/r_in;
AR = (b.*r)/(b_in*r_in);
x_outer = x - b/2.*sin(phi);
x_inner = x + b/2.*sin(phi);
r_outer = r + b/2.*cos(phi);
r_inner = r - b/2.*cos(phi);


%% Compute the pressure recovery factor
% Compressible definition (general)
Cp_compressible = (p-p_in)/(p0_in-p_in);

% Ideal pressure recovery coefficient
tan_a = v_t_in(1)/v_m_in(1);
Cp_ideal = 1 - (r_in./r).^2.*((b_in./b).^2 + tan_a^2)/(1 + tan_a^2);

% Store parameter distribution
solution.A = A;
solution.AR = AR;
solution.RR = RR;
solution.Cp = Cp_compressible;
solution.Cp_ideal = Cp_ideal;
solution.v = v;
solution.v_m = v_m;
solution.v_t = v_t;
solution.p = p;
solution.d = d;
solution.a = a;
solution.s = s;
solution.h = h;
solution.h0 = h0;
solution.s_gen = s_gen;
solution.Ma = Ma;
solution.Ma_t = Ma_t;
solution.Ma_m = Ma_m;
solution.m = m;
solution.r = r;
solution.x = x;
solution.b = b;
solution.theta = theta;
solution.theta_ideal = tan(alpha_in)*log(RR);
solution.x_outer = x_outer;
solution.x_inner = x_inner;
solution.r_outer = r_outer;
solution.r_inner = r_inner;

end


function dUdm = ode_diffuser(m,U,phi,div,r_in,b_in,x_in,Cf,fluid)

% Rename variables
v_m = U(1);
v_t = U(2);
d   = U(3);
p   = U(4);
alpha = atan(v_t/v_m);
v = sqrt(v_m^2+v_t^2);

% Local geometry
r = r_fun(r_in,phi,m);              % Radius as a function of m
x = x_fun(x_in,phi,m);              % Axial distance as a function of m
b = b_fun(b_in,div,m);              % Channel width as a function of m

% Derivative of the area change (forward finite differences)
delta = 1e-4;
diff_br = (b_fun(b_in,div,m+delta)*r_fun(r_in,phi,m+delta) - b*r)/delta;

% Derivative of internal energy with respect to pressure (constant density)
% e1 = PropsSI_scalar('U','P',p-delta,'D',d,fluid);
% e2 = PropsSI_scalar('U','P',p+delta,'D',d,fluid);
% dedp_d = (e2 - e1)/(2*delta);
dedp_d = PropsSI_scalar('d(U)/d(P)|D','P',p,'D',d,fluid);

% % Ideal gas limit (check)
% cp = PropsSI_scalar('CPMASS','P',p,'D',d,fluid);
% cv = PropsSI_scalar('CVMASS','P',p,'D',d,fluid);
% dedp_d_ideal = 1/(d*(cp/cv-1));

% Speed of sound (avoid computations in the two phase region)
a = PropsSI_scalar('A','P',p,'D',d,fluid);

% Stress at the wall
tau_w = Cf*d*v^2/2;        % Skin friction coefficient

% Heat flux at the wall
q_w = 0;                   % Adiabatic wall

% Coefficient matrix A
A = [d           0          v_m       0;
     d*v_m       0            0       1;
     0       d*v_m            0       0;
     0           0   -d*v_m*a^2   d*v_m];

% Source term vector
S = zeros(4,1);
S(1) = -d*v_m/(b*r)*diff_br;
S(2) = +d*v_t*v_t/r*sin(phi) - 2*tau_w/b*cos(alpha);
S(3) = -d*v_t*v_m/r*sin(phi) - 2*tau_w/b*sin(alpha);
S(4) = 2*(tau_w*v + q_w)/b/dedp_d;

% Obtain the slope of the solution by Gaussian elimination
dUdm = A\S;

% Check entropy generation
T = PropsSI_scalar('T','D',d,'P',p,fluid);
sigma = 2/b*(tau_w*v);
dUdm(5) = sigma/(d*v_m)/T;      % ds/dm
dUdm(6) = (v_t/v_m)/r;          % d(theta)/dr when phi=pi/2

end


function r = r_fun(r_in,phi,m)
r = r_in + sin(phi)*m;
end


function x = x_fun(x_in,phi,m)
x = x_in + cos(phi)*m;
end


function b = b_fun(b_in,div,m)
b = b_in + 2*tan(div)*m;
end


function [AR_check,isterminal,direction] = area_ratio(m,~,AR_prescribed,phi,div,r_in,b_in)

% Geometry
r = r_fun(r_in,phi,m);              % Radius as a function of m
b = b_fun(b_in,div,m);              % Channel width as a function of m
AR_current = (b*r)/(b_in*r_in);     % Current area ratio

% Stopping criterion
AR_check = AR_prescribed - AR_current;
isterminal = 1;   % stop the integration
direction = 0;    % negative direction

end


function p = static_pressure_from_mach(p0, T0, Ma, fluid)
    % Compute static pressure from stagnation state and Mach number
    s = PropsSI_scalar('S', 'P', p0, 'T', T0, fluid);
    h0 = PropsSI_scalar('H', 'P', p0, 'T', T0, fluid);
    p = fzero(@(p) stagnation_definition_error(p), p0);
    function res = stagnation_definition_error(p)
        a = PropsSI_scalar('A', 'P', p, 'S', s, fluid);
        h = PropsSI_scalar('H', 'P', p, 'S', s, fluid);
        v = a * Ma;
        res = h0 - h - v^2/2;
    end
end



%% Extensions of the code to make it more general:
% Use general functions for the geometry as input (instead of linear funcs)
% If the geometry is provided as general functions the local angle phi has
% to be computed by differentiation (finite differences)
% Perhaps prescribe the diffuser geometry using NURBS curves

% If the parametrization variable is not the meridional coordinate (m) then
% it is necessary to integrate the arclength to compute m

% Provide an arbitrary variation of the skin friction coefficient or use an
% empirical correlation to compute it

% Implement the heat transfer model in the code
% Perhaps use the Chilton-Colburn analogy to compute the heat transfer
% coefficient. It is also necessary to compute the stagnation temperature
% and to prescribe a wall temperature distribution



%% Compute the diffuser efficiency drop due to friction (deprecated)
% h01 = turbine_data.overall.h0_in;                                          % Stagnation enthalpy at the inlet of the turbine
% s1  = turbine_data.overall.s_in;                                           % Entropy at the inlet of the turbine
% h2s = turbine_data.overall.h_out_s;                                        % Static enthalpy at the outlet of the turbine for an isentropic expansion
% h_in_ss = PropsSI_scalar('H','P',p_in,'S',s1,fluid);                             % Static enthalpy at the inlet of the current cascade computed using the entropy at the inlet of the turbine
% h_out_ss = PropsSI_scalar('H','P',p_out,'S',s1,fluid);                           % Static enthalpy at the outlet of the current cascade computed using the entropy at the inlet of the turbine
% eta_drop_friction = ((h_in_ss-h_out_ss)-(h_in-h_out))/(h01-h2s);           % Efficiency drop due to skin friction (total-to-static efficiency)
% eta_drop_kinetic = (v_out^2/2)/(h01-h2s);                                  % Efficiency drop due to discharge kinetic energy (total-to-static efficiency)


