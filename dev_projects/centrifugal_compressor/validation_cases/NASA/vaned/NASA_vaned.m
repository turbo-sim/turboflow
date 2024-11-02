function pg = NASA_vaned()

% Geometry extracted from a .mat file, not verified with the original source

% Inducer geometry
pg.has_inducer = true;
pg.inducer_type = 'circular';
pg.L_inducer = 0.00;
% pg.D_0_s = 
% pg.D_0_h = 

% Inlet guide vanes
pg.has_igv = false;

% Impeller geometry
pg.D_1_s = 0.2100;
pg.D_1_h = 0.0820;
pg.D_2 = 0.4310;
pg.b_2 = 0.01700;
pg.L_z = 0.13224;
pg.Z_r_full = 15;
pg.Z_r_split = 0;
% pg.LR_split = 0;
pg.beta_1b_s = -54.8928;
pg.beta_1b_h = -32.7262;
pg.beta_1b = -43.8095;
pg.beta_2b = -46.0000;
pg.t_b_1 = 0.002644;
pg.t_b_2 = 0.007280;
pg.eps_a = 2.03e-04;
pg.eps_r = 2.03e-04;
pg.eps_b = 2.03e-04;
pg.Ra = 1.54e-06;

% Diffuser geometry
pg.has_vaned = true;
pg.vane_type = 'airfoil';
pg.Z_vd = 24;
pg.b_3 = 0.0167;
% pg.b_4 = 
% pg.b_5 = 
pg.D_3 = 0.46455;
pg.D_4 = 1.1*0.46455;
pg.D_5 = 0.67320;
pg.alpha_3b = 77.77;
pg.alpha_5b = 34.00;
pg.t_b_3 = 4.19e-04;
% pg.t_b_5 = 4.19e-04;

pg.AR_vd = 2.7540;
pg.AS_vd = 1.1257;

% Compressor outlet
pg.outlet_type = 'none';


end

