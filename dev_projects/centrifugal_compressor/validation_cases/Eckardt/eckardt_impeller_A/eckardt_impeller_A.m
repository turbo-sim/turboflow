function pg = eckardt_impeller_A()

% Geometry extracted from a .mat file, not verified with the original source

% Inducer geometry
pg.has_inducer = true;
pg.inducer_type = 'circular';
pg.L_inducer = 0.02;  % Guessed value
% pg.D_0_s = [];
% pg.D_0_h = [];

% Inlet guide vanes
pg.has_igv = false;

% Impeller geometry
pg.D_1_s = 0.280000000000000;
pg.D_1_h = 0.120000000000000;
pg.D_2 = 0.400000000000000;
pg.b_2 = 0.026000000000000;
pg.L_z = 0.130000000000000;
pg.Z_r_full = 20;
pg.Z_r_split = 0;
pg.beta_1b_s = -63.0;
pg.beta_1b_h = -40.0;
pg.beta_1b = -51.5;
pg.beta_2b = -30.0;
pg.t_b_1 = 0.002110000000000;
pg.t_b_2 = 0.001080000000000;
pg.eps_a = 2.350000000000000e-04;
pg.eps_r = 2.350000000000000e-04;
pg.eps_b = 2.350000000000000e-04;
pg.Ra = 2.000000000000000e-06;

% Diffuser geometry
pg.has_vanes = false;
pg.b_3 = 0.017302580000000;
pg.D_3 = 0.674800000000000;

% Compressor outlet
pg.outlet_type = 'none';

end

