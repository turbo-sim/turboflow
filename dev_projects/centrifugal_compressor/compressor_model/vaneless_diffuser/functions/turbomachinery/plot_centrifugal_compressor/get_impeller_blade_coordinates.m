%% Computation functions
% Documentation
% The function get_impeller_coordinates calculates the coordinates of the
% impeller in the z-r plane as a function of the (u,v) parameters
% The value v=0 corresponds to the hub and v=1 to the shroud

% The function get_impeller_arclength calculates the arclength of the
% meridional line at the bladespan specified by v
% The arclegnth is calculated integrating the arclength derivative with
% respect to the parameter u (ellipse angle) as an ODE equation
% The derivatives dr/du and dz/du are calculated with a central finite
% difference approximation

% The function get_impeller_wrap_angle calculates the wrapping angle
% distribution at the blade span specified by v
% The wrap angle distribution is calculated integrating the equation
% relating the wrap and metal angles as an ODE (system of two ODEs)
% The parameter to evaluate the metal angle is the local arclenght
% The blade metal angle is assumed to vary linearly from the leading to the
% trailing edge. A good parametrization should allow an arbitrary
% variation of blade metal angle along the blade streamwise direction

% The function get_impeller_thickness calculates the thickness distribution
% at the blade span specified by v
% The function requires the meridional length coordinate at the hub and the
% shroud. The first step is a scaling of the meridional length required for
% the case of splitted blades
% The current version of the meanline model does not support different
% thicknesses at the hub and the shroud, but this is not a big limitation
% The thickness distribution is very simple with a round leading edge and a
% round or clipped trailing edge. The blade thickness of the blade body is
% contant. A good parametrization should allow an arbitrary variation of
% blade thickness along the blade streamwise direction

% The function get_full_blade coordinates() calculates the coordinates of
% full blades at the hub and shroud sections
% In order to achieve zero rake angle at the exit, the wrapping angle at
% the shroud set to be equal to the wrapping angle at the hub (quick trick)

% Tge function get_splitter_blade_coordinates() calculates the coordinates
% of splitter blades at the hub and shroud sections
% The leading edge of the splittler blade is calculated approximately from
% the point of the full blades that is closes to the specified splittler
% blade length ration (found from discrete values, not from an exact
% computation involving the arclength of the full blade like an ODE
% with an event to stop the integration)
% The camber line of the full blades is re-used and rotation half the blade
% spacing angle.
% This implementation ensures that the splitter blade lies (almost) exactly
% in between 2 full blades
% The leading edge of the splitter blades is round thanks to the trick
% implemented in the thickness distribution function


function coords = get_impeller_blade_coordinates(geom, nameValueArgs)

    arguments
        geom (1, 1) struct
        nameValueArgs.N_points (1, 1) double = 300;
    end

    % Define parameter range
    u = [linspace(3*pi/2, 2*pi, nameValueArgs.N_points/2)];
    v = linspace(0, 1, 2);
    
    % Compute x-z coordinates of hub and shroud
    [z_h, r_h] = get_impeller_channel_coordinates(u, 0, geom);
    [z_s, r_s] = get_impeller_channel_coordinates(u, 1, geom);
    
    % Compute wrapping angle distribution
    [m_h, m_prime_h, theta_h, geom] = get_impeller_wrap_angle(u, 0, geom, 0.0, 1.0);
    [m_s, m_prime_s, theta_s, geom] = get_impeller_wrap_angle(u, 1, geom, 0.0, 1.0);
    theta_s = theta_h;
    
    % Compute camber surface hub
    x = (r_h.*cos(theta_h))';
    y = (r_h.*sin(theta_h))';
    z = z_h';
    xyz_c_h = [x(:), y(:), z(:)]';
    
    % Compute camber surface shroud
    x = (r_s.*cos(theta_s))';
    y = (r_s.*sin(theta_s))';
    z = z_s';
    xyz_c_s = [x(:), y(:), z(:)]';
    
    % Compute suction side hub
    t = get_impeller_thickness(m_h, m_s, 0.0, geom);
    x = r_h.*cos(theta_h + t/2./r_h);
    y = r_h.*sin(theta_h + t/2./r_h);
    z = z_h;
    xyz_ss_h = [x(:), y(:), z(:)]';
    theta_ss_h = theta_h + t/2./r_h;
    
    % Compute pressure side hub
    x = r_h.*cos(theta_h - t/2./r_h);
    y = r_h.*sin(theta_h - t/2./r_h);
    z = z_h;
    xyz_ps_h = [x(:), y(:), z(:)]';
    theta_ps_h = theta_h - t/2./r_h;
    
    % Compute suction side shroud
    t = get_impeller_thickness(m_h, m_s, 1.0, geom);
    x = r_s.*cos(theta_s + t/2./r_s);
    y = r_s.*sin(theta_s + t/2./r_s);
    z = z_s;
    xyz_ss_s = [x(:), y(:), z(:)]';
    theta_ss_s = theta_s + t/2./r_s;
    
    % Compute pressure side shroud
    x = r_s.*cos(theta_s - t/2./r_s);
    y = r_s.*sin(theta_s - t/2./r_s);
    z = z_s;
    xyz_ps_s = [x(:), y(:), z(:)]';
    theta_ps_s = theta_s - t/2./r_s;
    
    % Store coordinates in structure
    coords.m_h = m_h;
    coords.m_s = m_s;
    coords.m_prime_h = m_prime_h;
    coords.m_prime_s = m_prime_s;
    coords.theta_h = theta_h;
    coords.theta_s = theta_s;
    coords.theta_ss_h = theta_ss_h;
    coords.theta_ps_h = theta_ps_h;
    coords.theta_ss_s = theta_ss_s;
    coords.theta_ps_s = theta_ps_s;
    coords.z_h = z_h;
    coords.r_h = r_h;
    coords.z_s = z_s;
    coords.r_s = r_s;
    coords.xyz_c_h = xyz_c_h;
    coords.xyz_c_s = xyz_c_s;
    coords.xyz_ss_h = xyz_ss_h;
    coords.xyz_ps_h = xyz_ps_h;
    coords.xyz_ss_s = xyz_ss_s;
    coords.xyz_ps_s = xyz_ps_s;

end