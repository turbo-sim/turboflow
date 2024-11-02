function props = compute_properties_metastable_Td(T, d, fluid)

    % Compute the thermodynamic properties of a fluid using the Helmholtz
    % energy equation of state. All properties thermodynamic properties can
    % be derived as combinations of the Helmholtz energy and its
    % derivatives with respect to density and pressure
    %
    % This function can be used to estimate metastable properties using the
    % equation of state beyond the saturation lines
    % 

    % Update thermodynamic state
    fluid.update(py.CoolProp.CoolProp.DmassT_INPUTS, d, T)
    
    % Get fluid constant properties
    R = fluid.gas_constant;
    M = fluid.molar_mass;
    d_crit = fluid.rhomass_critical;
    T_crit = fluid.T_critical;

    % Compute reduced variables
    tau = T_crit/T;
    delta = d/d_crit;

    % Compute from the Helmholtz energy derivatives
    alpha = fluid.alpha0 + fluid.alphar;
    dalpha_dTau = fluid.dalpha0_dTau + fluid.dalphar_dTau;
    dalpha_dDelta = fluid.dalpha0_dDelta + fluid.dalphar_dDelta;
    d2alpha_dTau2 = fluid.d2alpha0_dTau2 + fluid.d2alphar_dTau2;
    d2alpha_dDelta2 = fluid.d2alpha0_dDelta2 + fluid.d2alphar_dDelta2;
    d2alpha_dDelta_dTau = fluid.d2alpha0_dDelta_dTau + fluid.d2alphar_dDelta_dTau;
    
    % Compute thermodynamic properties from Helmholtz energy EOS
    props = struct();
    props.T = T;
    props.p = (R/M)*T*d*delta*dalpha_dDelta;
    props.rhomass = d;
    props.umass = (R/M)*T*(tau*dalpha_dTau);
    props.hmass = (R/M)*T*(tau*dalpha_dTau+delta*dalpha_dDelta);
    props.smass = (R/M)*(tau*dalpha_dTau - alpha);
    props.gibbsmass = (R/M)*T*(alpha + delta*dalpha_dDelta);
    props.cvmass = (R/M)*(-tau^2*d2alpha_dTau2);
    props.cpmass = (R/M)*(-tau^2*d2alpha_dTau2 + (delta*dalpha_dDelta - delta*tau*d2alpha_dDelta_dTau)^2/(2*delta*dalpha_dDelta + delta^2*d2alpha_dDelta2));
    props.compressibility_factor = delta*dalpha_dDelta;
    props.isentropic_bulk_modulus = d*(R/M)*T*(2*delta*dalpha_dDelta + delta^2*d2alpha_dDelta2 - (delta*dalpha_dDelta - delta*tau*d2alpha_dDelta_dTau)^2/(tau^2*d2alpha_dTau2));
    props.speed_sound = sqrt(props.isentropic_bulk_modulus/d);
    props.isothermal_bulk_modulus = (R/M)*T*d*(2*delta*dalpha_dDelta + delta^2*d2alpha_dDelta2);
    props.isothermal_compressibility = 1/props.isothermal_bulk_modulus;
    props.isobaric_expansion_coefficient = 1/T*(delta*dalpha_dDelta - delta*tau*d2alpha_dDelta_dTau)/(2*delta*dalpha_dDelta + delta^2*d2alpha_dDelta2);
    props.viscosity = fluid.viscosity;
    props.conductivity = fluid.conductivity;

end