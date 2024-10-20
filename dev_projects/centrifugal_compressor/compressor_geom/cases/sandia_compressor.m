function pg = sandia_compressor()

% Inducer geometry
pg.has_inducer = true;
pg.inducer_type = 'flat';
pg.L_inducer = 0.75*(9.37-2.53)/1000;  % Estimated from figure
pg.D_0_s = 2*9.372047/1000;  % Estimated from photo
pg.D_0_h = 2*2.537585/1000;  % Estimated from photo

% Inlet guide vanes
pg.has_igv = false;

% Impeller geometry
pg.D_1_s = 2*9.372047/1000;     % From SANDIA report
pg.D_1_h = 2*2.537585/1000;     % From SANDIA report
pg.D_2 = 2*18.6817/1000;        % From SANDIA report
pg.b_2 =  1.7120/1000;          % From SANDIA report
% pg.L_z = 0.01137;             % Meroni (2018)
pg.L_z = 13.00/1000;            % Roberto webplotdigitizer
pg.Z_r_full = 6;                % Counted from photo
pg.Z_r_split = 6;               % Counted from photo
% pg.LR_split = 0.7;            % Meroni (2018)
pg.LR_split = 0.52;             % Roberto webplotdigitizer
pg.beta_1b_s = -50.00;          % From SANDIA report
pg.beta_1b_h = -17.88;          % Computed from tan(beta_1)/r_1 = cte
pg.beta_1b= -37.13;             % Computed from tan(beta_1)/r_1 = cte
pg.beta_2b = -50.00;            % From SANDIA report
pg.t_b_1 = 0.762/1000;          % From SANDIA report
pg.t_b_2 = 0.762/1000;          % From SANDIA report
pg.eps_a = 0.254/1000;          % From SANDIA report
pg.eps_r = 0.254/1000;          % From SANDIA report
pg.eps_b = 0.254/1000;          % From SANDIA report
pg.Ra = 10e-6;                  % Meroni (2018)

% Diffuser geometry
pg.has_vaned = true;            % From SANDIA report
% pg.has_vaned = false;            % From SANDIA report
pg.vane_type = 'wedge';         % From SANDIA report
pg.Z_vd = 17;                   % Counted from photo
pg.b_3 = pg.b_2;                % Meroni (2018)
pg.b_4 = pg.b_2;                % Meroni (2018)
pg.b_5 = pg.b_2;                % Meroni (2018)
% pg.D_3 = 1.02*pg.D_2;         % Meroni (2018)
% pg.D_4 = 1.02*pg.D_2;         % Meroni (2018)
% pg.D_5 = 1.37*pg.D_2;         % Meroni (2018)
% pg.D_5 = 1.687*pg.D_2;        % Meroni (2018) claims experimental data reffered to this section
pg.D_3 = (18.6817+0.3)*2/1000;  % Pecnik (2013) | Agrees with webplotdigitizer
% pg.D_4 = [];                  % D_4 is a dependent variable for wedge diffuser
pg.D_5 = 1.85*pg.D_2;           % Roberto webplotdigitizer script
% pg.alpha_3b = 71.5;           % From SANDIA report
pg.alpha_3b = 78;               % Trial and error to match w_in/r_2 measured from digitized figure
% pg.alpha_5b = [];             % alpha_5b is a dependent variable for wedge diffuser
pg.wedge_angle = 13.25;         % Roberto webplotdigitizer script
pg.t_b_3 = 0.00/1000;           % Assumed value for sharp edge
% pg.t_b_5 = 0.00/1000;         % t_b_5 is a dependenr variable for wedge diffuser

% According to Meroni (2018):
% IMPORTANT NOTE: THE DIFFUSER LENGTH IS r_3/r_2 = 2 BUT 
% THE EXPERIMENTAL DATA ARE REFERRED TO r_3/r_2 = 1.687

% Compressor outlet
pg.outlet_type = 'none';


end

