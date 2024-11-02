function options = options_compr(corrset)
if nargin==0
    corrset='default';
end
%%%%%%%%%%%%%%%%%%% GEOMETRY %%%%%%%%%%%%%%%%%%%%
options.noninteger_blade_warnings=true;
options.inadequate_geometry_warnings=true;
%%%%%%%%%%%%%%%%%%% MEAN-LINE %%%%%%%%%%%%%%%%%%%
options.findchoke=false;
options.oneshot=false;
options.stop_if_negative_opermargin=true;

options.solver='fsolve';
options.tolerance=1e-6;
options.step_tolerance=1e-9;
options.optimality_tolerance=0;
options.relaxed_tolerance=1e-4;

%%%%%%%%%%%%%%%%%%% COMPONENTS %%%%%%%%%%%%%%%%%%%
%%% compr_inducer
% calibration parameters
options.compr_inducer.blockage=0;


%%% compr_igv
% calibration parameters
options.compr_igv.blockage=0;
% loss models
options.compr_igv.loss_model='default';

%%% compr_rotor
% calibration parameters
options.compr_rotor.blockage=0;
options.compr_rotor.incidence_loss_fraction=0.6;
options.compr_rotor.clearance_loss_multiplier=1;
options.compr_rotor.Jhonston_mixing_loss_wake_mass_fraction=0.15;
% slip and loss models
options.compr_rotor.slip_model='default';
options.compr_rotor.loss_impeller_incidence_model='default';
options.compr_rotor.friction_model='default';
options.compr_rotor.bladeload_model='default';
options.compr_rotor.clearance_model='default';
options.compr_rotor.mixing_model='default';
options.compr_rotor.hubshroud_model='default';
options.compr_rotor.choking_model='default';
options.compr_rotor.supercritical_model='default';
options.compr_rotor.discfriction_model='default';
options.compr_rotor.leakage_model='default';
options.compr_rotor.distortion_model='default';
options.compr_rotor.diffusion_model='default';
options.compr_rotor.shock_model='default';
%stall criteria
options.compr_rotor.stall_criterion='default';
%density ratio between outlet and inlet of rotor, to aid convergence
options.rotor_density_guess_multiplier=1.1;

%%% compr_vaneless
options.compr_vaneless.use_0D_model=false;
options.compr_vaneless.friction_reference_Reynolds=1.8e5;
options.compr_vaneless.blockage=0;
options.compr_vaneless.stall_criterion='default';

%%% compr_vaned
options.compr_vaned.choking='on';

%%%%%%%%%%%%%%%% TEMPERATURE AND STRESS %%%%%%%%%%%
options.stress_temperature.total_temperature_recovery_factor=0.85;
options.stress_temperature.core_temperature_index=0.5;
options.stress_temperature.unshrouded_stress_constant=0.33;
options.stress_temperature.shrouded_stress_constant=0.55;
options.stress_temperature.rhometal=1;
%%%%%%%%%%%%%%%%%%% MAP BUILDER %%%%%%%%%%%%%%%%%%%
options.map.N_points_speedline=18;
options.map.Display_fsolve_map_builder='None';
options.map.end_speedline_at_maximum_CR=true;
options.map.end_speedline_at_zero_impeller_opermargin=false;
options.map.end_speedline_at_zero_diffuser_opermargin=false;

%%%%%%%%%%%%%%%%%%% PREDEFINED SET OF CORRELATIONS %%%%%%%%%%%%%%%%%%%
switch corrset
    case {'new','default'}
        options.compr_rotor.recirculation_model='Coppage';
        options.path_length_angles='metal';
        options.compr_rotor.diffusion_model='Aungier';
        options.compr_rotor.friction_loss_coefficient=5.6/2;
        options.compr_vaneless.friction_coefficient=0.0080;
    case 'Meroni'
        options.compr_rotor.recirculation_model='Oh';
        options.path_length_angles='fluid';
        options.compr_rotor.diffusion_model='no';
        options.compr_rotor.friction_loss_coefficient=4;
        options.compr_vaneless.friction_coefficient=0.0050;
    otherwise
        error('Unrecongized correlation set given as argument to options_compr')
end
end

