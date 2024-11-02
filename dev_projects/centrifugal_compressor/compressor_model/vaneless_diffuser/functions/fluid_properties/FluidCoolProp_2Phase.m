classdef FluidCoolProp_2Phase < handle
    
    % Alternative version of the FluidCoolProp class with some improvements
    %
    % - set_prop() functions refactored to reduce repeated boilerplate code
    % - get_flow() function refactored to save property list
    % - get_property() function refactores to avoid switch statement with
    %   each property stated explicitly + improved error message 
    % - Added functionality to compute mixed out conductivity and viscosity
    %   in the two phase region (volume weighted average)
    % - Added functionality to compute specific heat capacities in the two
    %   phase region (mass weighted average)
    % - Added functionality to compute the speed of sound in the two phase
    %   region. 2 models are available:
    %       1. Homogeneous equilibrium model (HEM)
    %       2. Homogeneous frozen model (HFM)
    %
    % Author: roagr
    % Date: 12.06.2023
    %
    
    properties
        abstractstate
        CoolProp
        converged_flag = false;
        throw_exceptions = true;
        backend
        fluid_name
        fluid_properties
    end

    methods

        % Class constructor
        function obj = FluidCoolProp_2Phase(backend,fluid_name)
            obj.backend = backend;
            obj.fluid_name = fluid_name;
            obj.CoolProp = py.importlib.import_module('CoolProp.CoolProp');
            obj.abstractstate = obj.CoolProp.AbstractState(backend,fluid_name);
        end
        
        % Set independent properties and compute dependent properties
        function set_prop(obj, input_type, prop_1, prop_2)
            try
                obj.fluid_properties = obj.compute_properties(input_type, prop_1, prop_2);
                obj.converged_flag = true;
            catch ME
                obj.converged_flag = false;
                if obj.throw_exceptions
                    throw(ME)
                end
            end
        end
       

        function props = compute_properties(obj, input_type, prop_1, prop_2)
            
            % Update Coolprop thermodynamic state
            obj.abstractstate.update(input_type, prop_1, prop_2);

            % Retrieve single-phase properties
            if obj.abstractstate.phase ~= obj.CoolProp.iphase_twophase

                % Get single-phase properties from CoolProp
                props.T = obj.abstractstate.T;
                props.p = obj.abstractstate.p;
                props.P = obj.abstractstate.p;
                props.rho = obj.abstractstate.rhomass;
                props.u = obj.abstractstate.hmass;
                props.h = obj.abstractstate.hmass;
                props.s = obj.abstractstate.smass;
                props.rhomass = obj.abstractstate.rhomass;
                props.umass = obj.abstractstate.umass;
                props.hmass = obj.abstractstate.hmass;
                props.smass = obj.abstractstate.smass;
                props.cp = obj.abstractstate.cpmass;
                props.cv = obj.abstractstate.cvmass;
                props.cpmass = obj.abstractstate.cpmass;
                props.cvmass = obj.abstractstate.cvmass;
                props.Z = obj.abstractstate.compressibility_factor;
                props.compressibility_factor = obj.abstractstate.compressibility_factor;
                props.isentropic_bulk_modulus = obj.abstractstate.rhomass*obj.abstractstate.speed_sound^2;
                props.speed_sound = obj.abstractstate.speed_sound;
                props.a = obj.abstractstate.speed_sound;
                props.a_HEM = props.a;
                props.a_HFM = props.a;
                props.isothermal_bulk_modulus = 1/obj.abstractstate.isothermal_compressibility;
                props.isothermal_compressibility = obj.abstractstate.isothermal_compressibility;
                props.isobaric_expansion_coefficient = obj.abstractstate.isobaric_expansion_coefficient;
                props.isentropic_expansion_coefficient = obj.abstractstate.rhomass/obj.abstractstate.p*obj.abstractstate.speed_sound^2;
                props.mu = obj.abstractstate.viscosity;
                props.k = obj.abstractstate.conductivity;
                props.viscosity = obj.abstractstate.viscosity;
                props.conductivity = obj.abstractstate.conductivity;
                props.Q = NaN;
                props.Qvol = NaN;

            % Compute two-phase properties
            else

                % Temperature and density of the two-phase mixture
                T_mix = obj.abstractstate.T;
                P_mix = obj.abstractstate.p;
                rho_mix = obj.abstractstate.rhomass;
                u_mix = obj.abstractstate.umass;
                h_mix = obj.abstractstate.hmass;
                s_mix = obj.abstractstate.smass;

                % Instantiane new fluid object to compute saturation
                % properties without changing the state of the class
                temp = obj.CoolProp.AbstractState(obj.backend, obj.fluid_name);

                % Saturated liquid properties
                temp.update(obj.CoolProp.QT_INPUTS, 0.00, T_mix)
                rho_L = temp.rhomass;
                u_L = temp.umass;
                h_L = temp.hmass;
                s_L = temp.smass;
                a_L = temp.speed_sound;
                cp_L = temp.cpmass;
                cv_L = temp.cvmass;
                k_L = temp.conductivity;
                mu_L = temp.viscosity;
                dsdp_L = temp.first_saturation_deriv(obj.CoolProp.iSmass, obj.CoolProp.iP);
                                
                % Saturated vapor properties
                temp.update(obj.CoolProp.QT_INPUTS, 1.00, T_mix)
                rho_V = temp.rhomass;
                u_V = temp.umass;
                h_V = temp.hmass;
                s_V = temp.smass;
                a_V = temp.speed_sound;
                cp_V = temp.cpmass;
                cv_V = temp.cvmass;
                k_V = temp.conductivity;
                mu_V = temp.viscosity;
                dsdp_V = temp.first_saturation_deriv(obj.CoolProp.iSmass, obj.CoolProp.iP);
                
                % Volume fractions of vapor and liquid
                Qvol_V = (rho_mix - rho_L) / (rho_V - rho_L);
                Qvol_L = 1.00 - Qvol_V;

                % Mass fractions of vapor and liquid
                Q_V = (1/rho_mix - 1/rho_L) / (1/rho_V - 1/rho_L);
                Q_L = 1.00 - Q_V;

                % Speed of sound of the two-phase mixture
                mechanical_equilibrium = Qvol_L/(rho_L*a_L^2) + Qvol_V/(rho_V*a_V^2);
                thermal_equilibrium = T_mix*(Qvol_L*rho_L/cp_L*dsdp_L^2 + Qvol_V*rho_V/cp_V*dsdp_V^2);
                compressibility_HEM = mechanical_equilibrium + thermal_equilibrium;
                compressibility_HFM = mechanical_equilibrium;
                if Q_V < 1e-6  % Avoid discontinuity when Q_v=0
                    a_HEM = a_L;
                    a_HFM = a_L;
                elseif Q_V > 1.0 - 1e-6  % Avoid discontinuity when Q_v=1
                    a_HEM = a_V;
                    a_HFM = a_V;
                else
                    a_HEM = sqrt(1/rho_mix/compressibility_HEM);
                    a_HFM = sqrt(1/rho_mix/compressibility_HFM);    
                end

                % Heat capacities of the two-phase mixture
                cp_mix = Q_L*cp_L + Q_V*cp_V;
                cv_mix = Q_L*cv_L + Q_V*cv_V;

                % Transport properties of the two-phase mixture
                k_mix = Qvol_L*k_L + Qvol_V*k_V;
                mu_mix = Qvol_L*mu_L + Qvol_V*mu_V;
                
                % Compressibility factor of the two-phase mixture
                % The Coolprop Z computation fails within the two-phase region
                % because it is based on differentiation of the Helmholtz
                % energy function (unphysical wiggles in the two-phase region)
                M = obj.abstractstate.molar_mass;
                R = obj.abstractstate.gas_constant;
                Z_mix = P_mix./(rho_mix.*(R/M).*T_mix);

                % Store properties in structure
                props.T = T_mix;
                props.p = P_mix;
                props.P = P_mix;
                props.rho = rho_mix;
                props.rho_L = rho_L;
                props.rho_V = rho_V;
                props.u = u_mix;
                props.u_L = u_L;
                props.u_V = u_V;
                props.h = h_mix;
                props.h_L = h_L;
                props.h_V = h_V;
                props.s = s_mix;
                props.s_L = s_L;
                props.s_V = s_V;
                props.cp = cp_mix;
                props.cp_L = cp_L;
                props.cp_V = cp_V;
                props.cv = cv_mix;
                props.cv_L = cv_L;
                props.cv_V = cv_V;
                props.a = a_HEM;
                props.a_HEM = a_HEM;
                props.a_HFM = a_HFM;
                props.a_L = a_L;
                props.a_V = a_V;
                props.Z = Z_mix;
                props.k = k_mix;
                props.k_L = k_L;
                props.k_V = k_V;
                props.mu = mu_mix;
                props.mu_L = mu_L;
                props.mu_V = mu_V;
                props.viscosity = mu_mix;
                props.conductivity = k_mix;
                props.Q = Q_V;
                props.Qvol = Qvol_V;
                props.isentropic_expansion_coefficient = rho_mix/P_mix*a_HEM^2;
                props.rhomass = rho_mix;
                props.umass = u_mix;
                props.hmass = h_mix;
                props.smass = s_mix;
                props.cvmass = cv_mix;
                props.cpmass = cp_mix;
                props.compressibility_factor = Z_mix;
                props.isentropic_bulk_modulus = rho_mix*a_HEM^2;
                props.speed_sound = a_HEM;
                props.isothermal_bulk_modulus = NaN;
                props.isothermal_compressibility = NaN;
                props.isobaric_expansion_coefficient = NaN;   
            end

        end

        % Get properties by string name
        function x = get_property(obj, propname)
            % Get property value dynamically
            try
                if isfield(obj.fluid_properties, propname)
                    x = obj.fluid_properties.(propname);
                else % Raise error message if the propname does not exist
                    options = "";
                    valid_options = fieldnames(obj.fluid_properties);
                    for i = 1:numel(valid_options)
                        options = strcat(options, "\t- ", valid_options(i), "\n");
                    end
                    error("The requested property '%s' is not available. The valid options are:\n%s", propname, sprintf(options))
                end

            % Always throw exception if propname does not exist
            catch ME       
                if obj.throw_exceptions || contains(ME.message, 'The requested property')
                    throw(ME)
                end
                x = NaN;
            end

        end


        % Get properties as a flow struct.
        % User must specify the additional fields of the flow struct, 
        % that are the mass flow rate and absolute velocity components
        function flow = get_flow(obj, G, Vm, Vt)
        
            % Define list of desired properties manually
            names = {'a', 'a_HEM', 'a_HFM', 'rho', 'T', 'P', 'h', 's', 'mu', 'k', 'cp', 'cv', ...
                     'isentropic_expansion_coefficient', 'Z'};
            
            % Store thermodynamic properties into flow structure
            if obj.converged_flag
                for i = 1:numel(names)
                    flow.(names{i}) = obj.fluid_properties.(names{i});
                end
            else
                for i = 1:numel(names)
                    flow.(names{i}) = NaN;
                end
            end
            
            % Store kinematic variables
            flow.G = G;
            flow.Vm = Vm;
            flow.Vt = Vt;

        end


        % Wrapper around different CoolProp inputs (in alphabetical order)
        function set_prop_h_p(obj, h, p)
            obj.set_prop(obj.CoolProp.HmassP_INPUTS, h, p)
        end

        function set_prop_h_Q(obj, h, Q)
            obj.set_prop(obj.CoolProp.HmassQ_INPUTS, h, Q)
        end

        function set_prop_hs(obj, h, s)
            obj.set_prop(obj.CoolProp.HmassSmass_INPUTS, h, s)
        end

        function set_prop_h_T(obj, h, T)
            obj.set_prop(obj.CoolProp.HmassT_INPUTS, h, T)
        end

        function set_prop_p_Q(obj, p, Q)
            obj.set_prop(obj.CoolProp.PQ_INPUTS, p, Q)
        end

        function set_prop_Ps(obj, p, s)
            obj.set_prop(obj.CoolProp.PSmass_INPUTS, p, s)
        end

        function set_prop_PT(obj, p, T)
            obj.set_prop(obj.CoolProp.PT_INPUTS, p, T)
        end

        function set_prop_p_u(obj, p, u)
            obj.set_prop(obj.CoolProp.PU_mass_INPUTS, p, u)
        end

        function set_prop_Q_s(obj, Q, s)
            obj.set_prop(obj.CoolProp.QSmass_INPUTS, Q, s)
        end   

        function set_prop_Q_T(obj, Q, T)
            obj.set_prop(obj.CoolProp.QT_INPUTS, Q, T)
        end   

        function set_prop_rhoh(obj, rho, h)
            obj.set_prop(obj.CoolProp.DmassHmass_INPUTS, rho, h)
        end

        function set_prop_rhoP(obj, rho, p)
            obj.set_prop(obj.CoolProp.DmassP_INPUTS, rho, p)
        end

        function set_prop_rho_Q(obj, rho, Q)
            obj.set_prop(obj.CoolProp.DmassQ_INPUTS, rho, Q)
        end

        function set_prop_rhos(obj, rho, s)
            obj.set_prop(obj.CoolProp.DmassSmass_INPUTS, rho, s)
        end   

        function set_prop_rho_T(obj, rho, T)
            obj.set_prop(obj.CoolProp.DmassT_INPUTS, rho, T)
        end

        function set_prop_rho_u(obj, rho, u)
            obj.set_prop(obj.CoolProp.DmassUmass_INPUTS, rho, u)
        end

        function set_prop_s_T(obj, s, T)
            obj.set_prop(obj.CoolProp.SmassT_INPUTS, s, T)
        end

        function set_prop_s_u(obj, s, u)
            obj.set_prop(obj.CoolProp.SmassUmass_INPUTS, s, u)
        end


    end  % End of methods


end  % End of class
