function out = PropsSI_scalar(varargin)
    
    % Wrapper around CoolProp high-level interface
    % This function accepts scalars as input variables

    % Calculate trivial thermodynamic properties
    if nargin == 2
        
        % Call Coolprop wrapper
        out = PropsSI_wrapper(varargin{:});
    
    % Calculate non-trivial thermodynamic properties
    elseif nargin == 6
        
        % Rename arguments
        name_out = varargin{1};

        % Speed of sound with Homogeneous Frozen Model
        if strcmp(name_out, 'A_HFM')
            out = compute_speed_of_sound_HFM(varargin{:});

        % Speed of sound with Homogeneous Equilibrium Model
        elseif strcmp(name_out, 'A_HEM')
            out = compute_speed_of_sound_HEM(varargin{:});
        
        % Speed of sound with smoothed Homogeneous Equilibrium Model
        elseif strcmp(name_out, 'A_HEMs')
            out = compute_speed_of_sound_HEMs(varargin{:});
                
        % Speed of sound with Finite Difference Method
        elseif strcmp(name_out, 'A_HEMcore')
            out = compute_speed_of_sound_HEM_core(varargin{:});
        
        % Speed of sound with Finite Difference Method
        elseif strcmp(name_out, 'A_FDM')
            out = compute_speed_of_sound_FDM(varargin{:});

        % Compressibility factor according to its definition
        elseif strcmp(name_out, 'Z_def')  
            out = compute_compressibility_factor(varargin{:});
            
        % Vapor quality based on mass
        elseif strcmp(name_out, 'Qmass')
            out = compute_vapor_quality_mass(varargin{:});

        % Vapor quality based on volume
        elseif strcmp(name_out, 'Qvol')
            out = compute_vapor_quality_volume(varargin{:});

        % Isobaric expansion coefficient with HFM
        elseif strcmp(name_out, 'isobaric_expansion_2phase')
            out = compute_isobaric_expansion(varargin{:});

        % Isobaric expansion coefficient with HFM
        elseif strcmp(name_out, 'isothermal_compressibility_2phase')
            out = compute_isothermal_compressibility(varargin{:});

        % All other thermodynamic properties
        else              
            out = PropsSI_wrapper(varargin{:});
        
        end

    
    % Invalid number of arguments
    else
        
        errID = 'thermodynamicEvaluation:inputError';
        msgtext = 'The number of arguments must be 2 or 6';
        throw(MException(errID,msgtext));

    end

end


%% Speed of sound models
function out = compute_speed_of_sound_HFM(varargin)

    % Use HFM model within two-phase region (piecewise function)
    if PropsSI_wrapper('Phase', varargin{2:end}) == 6  % inside two-phase region

        % Temperature and density of the two-phase mixture
        fluid_name = varargin{6};
        T_mix = PropsSI_wrapper('T', varargin{2:end});
        d_mix = PropsSI_wrapper('D', varargin{2:end});

        % Saturation properties
        d_liq = PropsSI_wrapper('D', 'T', T_mix, 'Q', 0.0, fluid_name);
        d_vap = PropsSI_wrapper('D', 'T', T_mix, 'Q', 1.0, fluid_name);
        a_liq = PropsSI_wrapper('A', 'T', T_mix, 'Q', 0.0, fluid_name);
        a_vap = PropsSI_wrapper('A', 'T', T_mix, 'Q', 1.0, fluid_name);

        % Volume fractions of vapor and liquid
        vfrac_vap = (d_mix - d_liq) / (d_vap - d_liq);
        vfrac_liq = 1.00 - vfrac_vap;
        
        % Speed of sound of the two-phase mixture
        compressibility_mix = vfrac_liq/(d_liq*a_liq^2) + vfrac_vap/(d_vap*a_vap^2);
        out = sqrt(1/d_mix/compressibility_mix);

    else  % outside two-phase region
        
        % Regular call to CoolProp
        out = PropsSI_wrapper('A', varargin{2:6});
        
    end

end


function out = compute_speed_of_sound_HEM(varargin)
    
    % Use HEM model within two-phase region (piecewise function)
    if PropsSI_wrapper('Phase', varargin{2:end}) == 6  % inside two-phase region
        out = compute_speed_of_sound_HEM_core(varargin{:});
    else  % outside two-phase region
        out = PropsSI_wrapper('A', varargin{2:end});
    end

end


function out = compute_speed_of_sound_HEMs(varargin)
    
    % Temperature and density of the two-phase mixture
    fluid_name = varargin{6};
    T_mix = PropsSI_wrapper('T', varargin{2:end});
    d_mix = PropsSI_wrapper('D', varargin{2:end});

    if T_mix < PropsSI_wrapper('T_critical', fluid_name)

        % Saturation properties
        d_liq = PropsSI_wrapper('D', 'T', T_mix, 'Q', 0.0, fluid_name);
        d_vap = PropsSI_wrapper('D', 'T', T_mix, 'Q', 1.0, fluid_name);

        % Speed of sound according to HFM in the single-phase region
        % Note that we need a function that can be evaluated inside and
        % outside the two-phase region to be able to do the blending
        fun_1phase = @(d) compute_speed_of_sound_HFM('A', 'T', T_mix, 'D', d, fluid_name);

        % Speed of sound according to HEM in the two-phase region
        fun_2phase = @(d) compute_speed_of_sound_HEM_core('A', 'T', T_mix, 'D', d, fluid_name);
        
        % Create blended function for vapor phase transition
        alpha_vap = 1/2;
        fun_blended = create_blended_function(fun_1phase, fun_2phase, d_vap, alpha_vap);

        % Create_blended_function for liquid phase transition
        alpha_liq = 1/2;
        fun_blended = create_blended_function(fun_blended, fun_1phase, d_liq, alpha_liq);
        
        % Evaluate smoothed HEM model
        out = fun_blended(d_mix);

    else
        
        out = PropsSI_wrapper('A', varargin{2:end});

    end

end


function out = compute_speed_of_sound_HEM_core(varargin)

    % Temperature and density of the two-phase mixture
    fluid_name = varargin{6};
    T_mix = PropsSI_wrapper('T', varargin{2:end});
    d_mix = PropsSI_wrapper('D', varargin{2:end});

    if T_mix < PropsSI_wrapper('T_critical', fluid_name)

        % Saturation properties
        d_liq = PropsSI_wrapper('D', 'T', T_mix, 'Q', 0.0, fluid_name);
        d_vap = PropsSI_wrapper('D', 'T', T_mix, 'Q', 1.0, fluid_name);
        a_liq = PropsSI_wrapper('A', 'T', T_mix, 'Q', 0.0, fluid_name);
        a_vap = PropsSI_wrapper('A', 'T', T_mix, 'Q', 1.0, fluid_name);
        cp_liq = PropsSI_wrapper('CPMASS', 'T', T_mix, 'Q', 0.0, fluid_name);
        cp_vap = PropsSI_wrapper('CPMASS', 'T', T_mix, 'Q', 1.0, fluid_name);
        dsdp_liq = PropsSI_wrapper('d(S)/d(P)|sigma', 'T', T_mix, 'Q', 0.0, fluid_name);
        dsdp_vap = PropsSI_wrapper('d(S)/d(P)|sigma', 'T', T_mix, 'Q', 1.0, fluid_name);

        % Volume fractions of vapor and liquid
        vfrac_vap = (d_mix - d_liq) / (d_vap - d_liq);
        vfrac_liq = 1.00 - vfrac_vap;
        
        % Speed of sound of the two-phase mixture
        mechanical_term = vfrac_liq/(d_liq*a_liq^2) + vfrac_vap/(d_vap*a_vap^2);
        equilibrium_term = T_mix*(vfrac_liq*d_liq/cp_liq*dsdp_liq^2 + vfrac_vap*d_vap/cp_vap*dsdp_vap^2);
        compressibility_mix = mechanical_term + equilibrium_term;
        out = sqrt(1/d_mix/compressibility_mix);

        % Speed of sound using Homogeneous Equilibrium Model (HEM)
        % This CoolProp calculation is wrong in the 2-phase region
        % value_out = sqrt(PropsSI_wrapper('d(P)/d(D)|S', varargin{2:end}));

    else
        
        out = PropsSI_wrapper('A', varargin{2:end});

    end

end


function out = compute_speed_of_sound_FDM(varargin)

    % Finite difference calculation
    fluid_name = varargin{6};
    s_mix = PropsSI_wrapper('S', varargin{2:end});
    d_mix = PropsSI_wrapper('D', varargin{2:end});
    eps = 1e-5*d_mix;
    p_plus = PropsSI_wrapper('P', 'D', d_mix+eps, 'S', s_mix, fluid_name);
    p_minus = PropsSI_wrapper('P', 'D', d_mix-eps, 'S', s_mix, fluid_name);
    out = sqrt((p_plus-p_minus)/(2*eps));

end



%% Vapor quality models
function out = compute_vapor_quality_mass(varargin)
    
    if PropsSI_wrapper('Phase', varargin{2:end}) == 6  % two-phase region

        % Temperature and density of the two-phase mixture
        fluid_name = varargin{6};
        T_mix = PropsSI_wrapper('T', varargin{2:end});
        d_mix = PropsSI_wrapper('D', varargin{2:end});
        d_liq = PropsSI_wrapper('D', 'T', T_mix, 'Q', 0.0, fluid_name);
        d_vap = PropsSI_wrapper('D', 'T', T_mix, 'Q', 1.0, fluid_name);

        % Volume fractions of vapor and liquid
        out = (1/d_mix - 1/d_liq) / (1/d_vap - 1/d_liq);

    else

        out = NaN;
        
    end

end

function out = compute_vapor_quality_volume(varargin)
          
    if PropsSI_wrapper('Phase', varargin{2:end}) == 6  % two-phase region

        % Temperature and density of the two-phase mixture
        fluid_name = varargin{6};
        T_mix = PropsSI_wrapper('T', varargin{2:end});
        d_mix = PropsSI_wrapper('D', varargin{2:end});
        d_liq = PropsSI_wrapper('D', 'T', T_mix, 'Q', 0.0, fluid_name);
        d_vap = PropsSI_wrapper('D', 'T', T_mix, 'Q', 1.0, fluid_name);

        % Volume fractions of vapor and liquid
        out = (d_mix - d_liq) / (d_vap - d_liq);

    else

        out = NaN;
        
    end

end



%% Compressibility factors
function out = compute_compressibility_factor(varargin)

    % Compressibility according to its definition
    fluid_name = varargin{6};
    M = PropsSI_wrapper('MOLAR_MASS', fluid_name);
    R = PropsSI_wrapper('GAS_CONSTANT', fluid_name);
    D = PropsSI_wrapper('D', varargin{2:end});
    T = PropsSI_wrapper('T', varargin{2:end});
    P = PropsSI_wrapper('P', varargin{2:end});
    out = P./(D.*(R/M).*T);

end


function out = compute_isobaric_expansion(varargin)

    % Use mixing model within two-phase region (piecewise function)
    if PropsSI_wrapper('Phase', varargin{2:end}) == 6  % inside two-phase region

        % Temperature and density of the two-phase mixture
        fluid_name = varargin{6};
        T_mix = PropsSI_wrapper('T', varargin{2:end});
        d_mix = PropsSI_wrapper('D', varargin{2:end});

        % Saturation properties
        d_liq = PropsSI_wrapper('D', 'T', T_mix, 'Q', 0.0, fluid_name);
        d_vap = PropsSI_wrapper('D', 'T', T_mix, 'Q', 1.0, fluid_name);
        beta_liq = PropsSI_wrapper('ISOBARIC_EXPANSION_COEFFICIENT', 'T', T_mix, 'Q', 0.0, fluid_name);
        beta_vap = PropsSI_wrapper('ISOBARIC_EXPANSION_COEFFICIENT', 'T', T_mix, 'Q', 1.0, fluid_name);

        % Volume fractions of vapor and liquid
        vfrac_vap = (d_mix - d_liq) / (d_vap - d_liq);
        vfrac_liq = 1.00 - vfrac_vap;
        
        % Speed of sound of the two-phase mixture
        out = vfrac_liq*beta_liq + vfrac_vap*beta_vap;

    else  % outside two-phase region
        
        % Regular call to CoolProp
        out = PropsSI_wrapper('ISOBARIC_EXPANSION_COEFFICIENT', varargin{2:6});
        
    end

end


function out = compute_isothermal_compressibility(varargin)

    % Use mixing model within two-phase region (piecewise function)
    if PropsSI_wrapper('Phase', varargin{2:end}) == 6  % inside two-phase region

        % Temperature and density of the two-phase mixture
        fluid_name = varargin{6};
        T_mix = PropsSI_wrapper('T', varargin{2:end});
        d_mix = PropsSI_wrapper('D', varargin{2:end});

        % Saturation properties
        d_liq = PropsSI_wrapper('D', 'T', T_mix, 'Q', 0.0, fluid_name);
        d_vap = PropsSI_wrapper('D', 'T', T_mix, 'Q', 1.0, fluid_name);
        beta_liq = PropsSI_wrapper('ISOTHERMAL_COMPRESSIBILITY', 'T', T_mix, 'Q', 0.0, fluid_name);
        beta_vap = PropsSI_wrapper('ISOTHERMAL_COMPRESSIBILITY', 'T', T_mix, 'Q', 1.0, fluid_name);

        % Volume fractions of vapor and liquid
        vfrac_vap = (d_mix - d_liq) / (d_vap - d_liq);
        vfrac_liq = 1.00 - vfrac_vap;
        
        % Speed of sound of the two-phase mixture
        out = vfrac_liq*beta_liq + vfrac_vap*beta_vap;

    else  % outside two-phase region
        
        % Regular call to CoolProp
        out = PropsSI_wrapper('ISOTHERMAL_COMPRESSIBILITY', varargin{2:6});
        
    end

end


%% Other functions
function value_out = PropsSI_wrapper(varargin)

    % Wrap the scalar Python function call to Coolprop
    value_out = py.CoolProp.CoolProp.PropsSI(varargin{:});

end


