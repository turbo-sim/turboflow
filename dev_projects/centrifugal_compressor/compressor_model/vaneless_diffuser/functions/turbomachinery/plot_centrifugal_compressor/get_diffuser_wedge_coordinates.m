function coords = get_diffuser_wedge_coordinates(geom, N_points)

    % Rename variables
    d_theta = 360/geom.Z_vd;
    phi = geom.alpha_3b;
    eps = geom.wedge_angle;
    w_in = geom.o_4;
    w_out = geom.o_5;
    lambda = (sqrt((geom.r_5/geom.r_3)^2 - sind(phi + eps/2)^2) - cosd(phi + eps/2));

    % Initial angle
    theta_0 = 0.00;
    x_0 = cosd(theta_0)*geom.r_3;
    y_0 = sind(theta_0)*geom.r_3;

    % Intersection of the the diffuser outer side with the exit radius
    % Analytic solution of easy second order equation
    angle_1 = theta_0 + geom.alpha_3b + geom.wedge_angle/2;
    dist_1 = sqrt(geom.r_5^2-geom.r_3^2*sind(angle_1)^2) - geom.r_3*cosd(angle_1);
    dist_1 = linspace(0, dist_1, floor(N_points/3))';
    x_1 = x_0 + dist_1*cosd(angle_1);
    y_1 = y_0 + dist_1*sind(angle_1);
    theta_1 = atan2(y_1(end), x_1(end));
    
    % Intersection of the the diffuser inner side with the exit radius
    angle_2 = theta_0 + geom.alpha_3b - geom.wedge_angle/2;
    dist_2 = sqrt(geom.r_5^2 - geom.r_3^2*sind(angle_2)^2) - geom.r_3*cosd(angle_2);
    dist_2 = linspace(0, dist_2, floor(N_points/3))';
    x_2 = x_0 + dist_2*cosd(angle_2);
    y_2 = y_0 + dist_2*sind(angle_2);
    theta_2 = atan2(y_2(end), x_2(end));
    
    % Exit radius arc
    phi_arc = linspace(theta_1, theta_2, ceil(N_points/3))';
    x_arc = geom.r_5*cos(phi_arc);
    y_arc = geom.r_5*sin(phi_arc);
    
    % Vaned diffuser hub coordinates    
    xyz_h = [  x_1,   y_1, geom.L_z+0*x_1
             x_arc, y_arc, geom.L_z+0*x_arc
            flip(x_2), flip(y_2), geom.L_z+0*x_1]';
    
    % Vaned diffuser shroud coordinates
    xyz_s = [  x_1,   y_1, geom.L_z-geom.b_3+0*x_1
             x_arc, y_arc, geom.L_z-geom.b_5+0*x_arc
               flip(x_2), flip(y_2), geom.L_z-geom.b_3+0*x_1]';

    % Compute channel coordinates
    X1_in = geom.r_3*cosd(theta_0+d_theta);
    Y1_in = geom.r_3*sind(theta_0+d_theta);
    X2_in = geom.r_3*cosd(theta_0+d_theta) + w_in*sind(theta_0+d_theta/2+phi);  
    Y2_in = geom.r_3*sind(theta_0+d_theta) - w_in*cosd(theta_0+d_theta/2+phi);
    X1_out = geom.r_3*cosd(theta_0) + geom.r_3*lambda*cosd(theta_0+phi+eps/2);
    Y1_out = geom.r_3*sind(theta_0) + geom.r_3*lambda*sind(theta_0+phi+eps/2);
    X2_out = geom.r_3*cosd(theta_0) + geom.r_3*lambda*cosd(theta_0+phi+eps/2) - w_out*sind(theta_0+d_theta/2+phi);  
    Y2_out = geom.r_3*sind(theta_0) + geom.r_3*lambda*sind(theta_0+phi+eps/2) + w_out*cosd(theta_0+d_theta/2+phi);

    % Organize coordinates
    xyz_channel = [X1_in, Y1_in, geom.L_z-geom.b_3 
                   X2_in, Y2_in, geom.L_z-geom.b_3
                   X1_out, Y1_out, geom.L_z-geom.b_5
                   X2_out, Y2_out, geom.L_z-geom.b_5]';

    xyz_inlet = [X1_in, Y1_in, geom.L_z 
                 X2_in, Y2_in, geom.L_z
                 X2_in, Y2_in, geom.L_z-geom.b_3
                 X1_in, Y1_in, geom.L_z-geom.b_3]';

    xyz_outlet = [X1_out, Y1_out, geom.L_z 
                  X2_out, Y2_out, geom.L_z
                  X2_out, Y2_out, geom.L_z-geom.b_5
                  X1_out, Y1_out, geom.L_z-geom.b_5]';


    % Process coordinates
    xyz = [xyz_h, xyz_s];

    % Store coordinates in structure
    coords.xyz = xyz;
    coords.xyz_h = xyz_h;
    coords.xyz_s = xyz_s;
    coords.xyz_channel = xyz_channel;
    coords.xyz_inlet = xyz_inlet;
    coords.xyz_outlet = xyz_outlet;

end