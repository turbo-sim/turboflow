function geom = complete_geometry_compr(PG,options)
% geom = complete_geometry_compr(PG,options)
% Completes the geometry of the centrifugal compressor, starting from
% some user defined parameters and options.
% The user defined parameters should be supplied as fields of the struct PG
% (partial geometry). A description of the parameters used is supplied in
% the additional documentation.
% If some required fields are missing an error will be thrown. If the PG
% struct contains unused field a warning is given (might have mispelled something).


%% Check presence of optional components
% Inlet guide vanes - Options: true, false
[has_igv,PG] = getfield_else_default(PG,'has_igv',false);

% Inducer - Options: true, false
% If has_inducer=false, then D_0=D_1
[has_inducer,PG] = getfield_else_default(PG,'has_inducer',false);

% Type of diffuser - Options: false (vaneless), true (vaned)
[has_vaned,PG] = getfield_else_default(PG,'has_vaned',false);

% Type of outlet - Options: none, volute, return channel
[outlet_type,PG] = getfield_else_default(PG,'outlet_type','none');


%% Rotor
% Shrouded rotor (extra centrifugal stress and windage)
[is_shrouded,PG] = getfield_else_default(PG, 'is_shrouded',false);

% Roughness (unique value for entire machine)
[Ra,PG] = getfield_else_default(PG,'Ra',5e-6);

% Inlet diameters
[D_1_h,PG] = getfield_else_default(PG,'D_1_h');
[D_1_s,PG] = getfield_else_default(PG,'D_1_s');
D_1 = sqrt((D_1_h^2+D_1_s^2)/2);
b_1 = (D_1_s-D_1_h)/2;

% Inlet blade angle
[beta_1b_s,PG] = getfield_else_default(PG,'beta_1b_s');
if isfield(PG,'beta_1b') && isfield(PG,'beta_1b_h')
    beta_1b = PG.beta_1b;
    beta_1b_h = PG.beta_1b_h;
    PG=rmfield(PG,'beta_1b');
    PG=rmfield(PG,'beta_1b_h');
else
    if isfield(PG,'beta_1b') || isfield(PG,'beta_1b_h')
        error('If rotor blade angle at mid or hub are specified, then both value must be inputed')
    end
    % Linear variation of tan(beta_1) with radius (alpha_1=zero)
    % The ideal spanwise distribution is more complex if alpha_1=/=zero
    tanb1s = tand(beta_1b_s);
    beta_1b = atand(tanb1s*D_1/D_1_s);
    beta_1b_h = atand(tanb1s*D_1_h/D_1_s);
end

% Inlet area (without blades)
if isfield(PG,'A_1_eff')
    A_1_eff = PG.A_1_eff;
    PG=rmfield(PG,'A_1_eff');
else
    if has_inducer
        B1 = options.compr_inducer.blockage;
    elseif has_igv
        B1 = options.compr_igv.blockage;
    else
        B1 = 0;
    end
    A_1_eff = (1-B1)*(pi/4)*(D_1_s^2-D_1_h^2);
end

% Inlet blade thickness
[t_b_1,PG] = getfield_else_default(PG,'t_b_1',max(0.2e-3,PG.D_2/200));

% Blade number
[Z_r_full,PG] = getfield_else_default(PG,'Z_r_full');
[Z_r_split,PG] = getfield_else_default(PG,'Z_r_split',0);
if Z_r_split>0
    [LR_split,PG] = getfield_else_default(PG,'LR_split',0.7);
else
    LR_split=0;
end
Z_r=Z_r_full+LR_split*Z_r_split;

% Throat area
[A_rth,PG] = getfield_else_default(PG,'A_rth',A_1_eff*cosd(beta_1b)-t_b_1*Z_r_full*b_1);

% Clearances
if isfield(PG,'clearance')
    PG.eps_a=PG.clearance;
    PG = rmfield(PG,'clearance');
end
[eps_a,PG] = getfield_else_default(PG,'eps_a',0.5e-3);
[eps_r,PG] = getfield_else_default(PG,'eps_r',eps_a);
[eps_b,PG] = getfield_else_default(PG,'eps_b',eps_a);

% Rotor outlet
[D_2,PG] = getfield_else_default(PG,'D_2');
[b_2,PG] = getfield_else_default(PG,'b_2');
[beta_2b,PG] = getfield_else_default(PG,'beta_2b');
[t_b_2,PG] = getfield_else_default(PG,'t_b_2',t_b_1);
B2=options.compr_rotor.blockage;
[A_2_eff,PG] = getfield_else_default(PG,'A_2_eff',pi*D_2*b_2*(1-B2));

% Axial lenght
[L_z,PG] = getfield_else_default(PG,'L_z',0.5*(D_2-D_1_s+b_2));


%% Inducer
if has_inducer
    % Inducer type only affects visualization, not computations
    [inducer_type,PG] = getfield_else_default(PG,'inducer_type');
    check_valid_options(inducer_type, {'flat', 'circular', 'elliptic'})
    [L_inducer,PG] = getfield_else_default(PG,'L_inducer');
    [D_0_h,PG] = getfield_else_default(PG,'D_0_h', D_1_h);
    [D_0_s,PG] = getfield_else_default(PG,'D_0_s', D_1_s);
    D_0 = sqrt((D_0_h^2+D_0_s^2)/2);
    A_0 = pi/4*(D_0_s^2-D_0_h^2);
    D_hyd_inducer = 0.5*((D_0_s-D_0_h)+(D_1_s-D_1_h));
else
    inducer_type = 'flat';
    L_inducer = 0.00;
    D_0_h = D_1_h;
    D_0_s = D_1_s;
    D_0 = D_1;
    A_0 = A_1_eff;
end


%% IGV
if has_igv
    [D_in_h,PG] = getfield_else_default(PG,'D_in_h');
    [D_in_s,PG] = getfield_else_default(PG,'D_in_s');
    [alpha_0b,PG] = getfield_else_default(PG,'alpha_0b');
    [Z_igv,PG] = getfield_else_default(PG,'Z_igv');
    D_in = sqrt((D_in_h^2+D_in_s^2)/2);
    A_in = pi/4*(D_in_s^2-D_in_h^2);
    [alpha_inb,PG] = getfield_else_default(PG,'alpha_inb',alpha_0b);
    [ac_igv,PG] = getfield_else_default(PG,'ac_igv',0.5);
    [c_igv,PG] = getfield_else_default(PG,'c_igv',pi*D_0/Z_igv);
else
    D_in_h = D_0_h;
    D_in = D_0;
    D_in_s = D_0_s;
    A_in = A_0;
    Z_igv = 0;
end


%% Vaneless diffuser
[D_3,PG] = getfield_else_default(PG,'D_3');
[b_3,PG] = getfield_else_default(PG,'b_3',b_2);
BF_3 = options.compr_vaneless.blockage;
[A_3_eff,PG] = getfield_else_default(PG,'A_3_eff',pi*D_3*b_3*(1-BF_3));
dbdr_vaneless=2*(b_3-b_2)/(D_3-D_2);


%% Vaned diffuser
if has_vaned

    % Type of vanes when has_vaned=true - Options: airfoil, wedge
    [vane_type,PG] = getfield_else_default(PG,'vane_type','airfoil');
    check_valid_options(vane_type, {'airfoil', 'wedge'})
    if strcmp(vane_type, 'airfoil')
        [b_4,PG] = getfield_else_default(PG,'b_4',b_3);
        [b_5,PG] = getfield_else_default(PG,'b_5',b_3);
        [D_4,PG] = getfield_else_default(PG,'D_4');
        [D_5,PG] = getfield_else_default(PG,'D_5');
        [t_b_3,PG] = getfield_else_default(PG,'t_b_3',t_b_2);
        [t_b_5,PG] = getfield_else_default(PG,'t_b_5',t_b_3);
        [Z_vd,PG]=getfield_else_default(PG,'Z_vd');
        [alpha_3b,PG] = getfield_else_default(PG,'alpha_3b');
        [alpha_5b,PG] = getfield_else_default(PG,'alpha_5b');
        [alpha_35m,PG] = getfield_else_default(PG,'alpha_35m',0.5*(alpha_5b+alpha_3b));
        ac_vd = (2-(alpha_35m-alpha_3b)/(alpha_5b-alpha_3b))/3;
        [L_b_vd,PG] = getfield_else_default(PG,'L_b_vd',(D_5-D_3)/(2*cosd(alpha_35m)));
        sigma_vd = (Z_vd*L_b_vd)/(pi*D_3);  %% error
        theta_vd = alpha_3b-alpha_5b;
        if abs(alpha_5b-alpha_3b)>0
            % Suggested by Aungier1990. In Aungier2000 a similar formula is
            % present but 0.02 is used instead of 0.002. This is most likely a
            % misprint since it would lead to unphysically high deviations
            delta0_vd = theta_vd*(0.92*ac_vd^2+0.002*alpha_5b)/(sigma_vd^0.5-0.002*theta_vd);
        else
            delta0_vd = 0;
        end
        if isfield(PG,'AS_vd')
            AS_vd = PG.AS_vd;
            o_4 = b_4*AS_vd;
            PG = rmfield(PG,'AS_vd');
        elseif isfield(PG,'o_4')
            o_4 = PG.o_4;
            AS_vd = o_4/b_4;
            PG = rmfield(PG,'o_4');
        else
            o_4 = pi*D_3*cosd(alpha_3b)/Z_vd-t_b_3;
            AS_vd = o_4/b_4;
        end
        if isfield(PG,'AR_vd')
            AR_vd = PG.AR_vd;
            o_5 = o_4*AR_vd;
            PG = rmfield(PG,'AR_vd');
        else
            o_5 = pi*D_5*cosd(alpha_5b)/Z_vd-t_b_5;
            AR_vd = (o_5*b_5)/(o_4*b_4);
        end
        A_th_4 = o_4*b_4*Z_vd;
        C_r_th_vd = min(1,(A_3_eff*cosd(alpha_3b)/A_th_4)^(0.5)); %throat contraction ratio
        A_th_4_eff = A_th_4*C_r_th_vd;
        alpha_th_4 = acosd(A_th_4/A_3_eff);
        D_hyd_vd = o_4*b_4/(o_4+b_4)+o_5*b_5/(o_5+b_5);
        LR_vd = L_b_vd/o_4;
        DdeltaDinc_vd = exp(-sigma_vd*(3.3-(alpha_3b/60)^2));
        A_5 = pi*D_5*b_5;

    elseif strcmp(vane_type, 'wedge')

        % Input variables
        [b_4,PG] = getfield_else_default(PG,'b_4',b_3);
        [b_5,PG] = getfield_else_default(PG,'b_5',b_3);
        [D_5,PG] = getfield_else_default(PG,'D_5');
        [t_b_3,PG] = getfield_else_default(PG,'t_b_3',0-00);
        [Z_vd,PG] = getfield_else_default(PG,'Z_vd');
        [alpha_3b,PG] = getfield_else_default(PG,'alpha_3b');
        [wedge_angle,PG] = getfield_else_default(PG,'wedge_angle');

        % Computed variables
        r_2 = D_2/2;
        r_3 = D_3/2;
        r_5 = D_5/2;
        phi = alpha_3b*pi/180;
        eps = wedge_angle*pi/180;
        d_theta = 2*pi/Z_vd;
        % div = (d_theta - eps);
        lambda = (sqrt((r_5/r_3)^2 - sin(phi + eps/2)^2) - cos(phi + eps/2));
        % gamma = acos((r_5^2-(lambda^2-1)*r_3^2)/(2*r_3*r_5));
        o_4 = r_3*(sin(phi+eps/2) - sin(phi + eps/2 - d_theta))/cos(d_theta/2 - eps/2);
        o_5 = r_3*(sin(d_theta+phi-eps/2) + lambda*sin(d_theta-eps) - sin(phi-eps/2))/cos(d_theta/2-eps/2);
        L_in = r_3*(cos(phi-d_theta/2) - cos(phi + d_theta/2))/(cos(d_theta/2 - eps/2));
        L_out = lambda*r_3 - L_in;
        % length = (o_5 - o_4)/(2*tan(div/2));
        length = L_out*cos((d_theta-eps)/2);
        r_4 = sqrt(r_3^2+(o_4/2)^2-r_3*o_4*sin(d_theta/2-phi));
        D_4 = 2*r_4;
        % r_out = sqrt(r_5^2+(o_5/2)^2-r_5*o_5*sin(d_theta/2+phi-gamma));
        alpha_5b = atand(sin(phi)./(sqrt((r_5/r_3).^2-sin(phi).^2)));
        wedge_length = sqrt((r_5/r_3)^2 - sin(phi)^2) - cos(phi);
        t_b_5 = 2*wedge_length*tan(eps/2);

        % Copy-pasted from airfoil diffuser (to be checked)
        ac_vd = [];
        alpha_35m = (alpha_5b + alpha_3b)/2;
        AR_vd = o_5/o_4;
        AS_vd = o_4/b_4;
        theta_vd = alpha_5b - alpha_3b;
        L_b_vd = length;
        sigma_vd = (Z_vd*L_b_vd)/(pi*D_3);
        delta0_vd = 0;
        A_th_4 = o_4*b_4*Z_vd;
        C_r_th_vd = min(1,(A_3_eff*cosd(alpha_3b)/A_th_4)^(0.5)); %throat contraction ratio
        A_th_4_eff = A_th_4*C_r_th_vd;
        alpha_th_4 = acosd(A_th_4/A_3_eff);
        D_hyd_vd = o_4*b_4/(o_4+b_4)+o_5*b_5/(o_5+b_5);
        LR_vd=L_b_vd/o_4;
        DdeltaDinc_vd = exp(-sigma_vd*(3.3-(alpha_3b/60)^2));
        A_5=pi*D_5*b_5;

    else
        error("Invalid vane type")

    end

else % no vanes
    Z_vd = 0;
    [D_4,PG] = getfield_else_default(PG,'D_4', D_3);
    [D_5,PG] = getfield_else_default(PG,'D_5', D_3);
    [b_4,PG] = getfield_else_default(PG,'b_4', b_3);
    [b_5,PG] = getfield_else_default(PG,'b_5', b_3);
end


%% Outlet channel
check_valid_options(outlet_type, {'volute', 'return channel', 'none'})
if strcmpi(outlet_type,'volute')
    [A_volute,PG]=getfield_else_default(PG,'A_volute');
    [A_exitcone,PG]=getfield_else_default(PG,'A_exitcone',A_volute);
    [volute_type,PG]=getfield_else_default(PG,'volute_type','asymmetric');
    if strcmpi(volute_type,'semiexternal')
        r_6=D_5/2;
    elseif strcmpi(volute_type,'symmetric')||strcmpi(volute_type,'asymmetric')||strcmpi(volute_type,'default')
        r_6=D_5/2+sqrt(A_volute/(0.75*pi+0.25)); %add the radius at the volute outlet
    else
        error('Volute type can be semiexternal, symmetric or asymmetric(default)')
    end
    [D_6,PG]=getfield_else_default(PG,'D_6',r_6*2);
    Z_rc=0;

elseif strcmpi(outlet_type,'return channel')
    D_6=D_5;
    [b_6,PG]=getfield_else_default(PG,'b_6');
    [alpha_6b,PG]=getfield_else_default(PG,'alpha_6b');
    [alpha_7b,PG]=getfield_else_default(PG,'alpha_7b');
    [R_56,PG]=getfield_else_default(PG,'R_56',(b_5+b_6)/2);
    [Z_rc,PG]=getfield_else_default(PG,'Z_rc');
    [b_7,PG]=getfield_else_default(PG,'b_7');
    [D_7,PG]=getfield_else_default(PG,'D_7');
    [D_8_s,PG]=getfield_else_default(PG,'D_8_s',D_7-2*b_7);
    [D_8_h,PG]=getfield_else_default(PG,'D_8_h',D_8_s-2*b_7);
    D_8=sqrt((D_8_h^2+D_8_s^2)/2);
    b_8=0.5*(D_8_s-D_8_h);
    A_8=pi/4*(D_8_s^2-D_8_h^2);
    [t_b_6,PG]=getfield_else_default(PG,'t_b_6',t_b_3);
    [alpha_67m,PG] = getfield_else_default(PG,'alpha_67m',0.5*(alpha_7b+alpha_6b));
    ac_rc = (2-(alpha_67m-alpha_6b)/(alpha_7b-alpha_6b))/3;
    A_7=pi*D_7*b_7;
    L_b_rc = abs(D_7-D_6)/(2*cosd(alpha_67m));
    sigma_rc=Z_rc*abs(D_7-D_6)/(2*pi*D_7*sind(abs(alpha_67m)));
    theta_rc=alpha_6b-alpha_7b;
    delta0_rc=theta_rc*(0.92*ac_rc^2+0.002*alpha_6b)/(sqrt(sigma_rc)-0.002*theta_rc);
    DdeltaDinc_rc=exp(-sigma_rc*(3.3-(alpha_6b/60)^2));
    o_6=pi*D_6*cosd(alpha_6b)/Z_rc-t_b_6;
    o_7=pi*D_7*cosd(alpha_7b)/Z_rc-t_b_6;
    D_hyd_rc = o_6*b_6/(o_6+b_6)+o_7*b_7/(o_7+b_7);

elseif strcmpi(outlet_type,'none')
    Z_rc=0;
    [D_6,PG]=getfield_else_default(PG,'D_6', D_5*1.1);
    [b_6,PG]=getfield_else_default(PG,'b_6', b_5);

else
    error('Invalid outlet_type')
end


%% Impeller backplate
[include_backface_geometry,PG] = getfield_else_default(PG,'include_backface_geometry',true);
if include_backface_geometry

%     % Original Simone
%     [t_impeller,PG] = getfield_else_default(PG,'t_impeller',0.1*b_2);
%     [t_backface,PG] = getfield_else_default(PG,'t_backface',0.06*D_2);
%     [D_backface,PG] = getfield_else_default(PG,'D_backface',1.2*D_1_h);
%     [D_shaft,PG] = getfield_else_default(PG,'D_shaft',0.5*D_1_h);
        
      % Casey and Robinson (2001) implemented by Roberto
      [backplate_type, PG] = getfield_else_default(PG,'backplate_type', 'circular_arc');
      check_valid_options(backplate_type, {'linear', 'circular_arc'});
      [t_backplate_1, PG] = getfield_else_default(PG,'t_backplate_1', 0.20*b_2);
      [t_backplate_2, PG] = getfield_else_default(PG,'t_backplate_2', 0.06*D_2);
      [D_backplate_1, PG] = getfield_else_default(PG,'r_backplate_1', 1.00*D_2);
      [D_backplate_2, PG] = getfield_else_default(PG,'r_backplate_2', 1.25*D_1_h);
      [D_shaft, PG] = getfield_else_default(PG,'D_shaft',0.5*D_1_h);
      r_backplate_1 = 0.5*D_backplate_1;
      r_backplate_2 = 0.5*D_backplate_2;
      r_shaft = 0.5*D_shaft;
      L_shaft = L_z/3;  % Only visual
      if strcmp(backplate_type, 'circular_arc')
          r_min = sqrt((t_backplate_2 - t_backplate_1)^2 + (r_backplate_2 - r_backplate_1)^2)/2;
          [R_backplate, PG] = getfield_else_default(PG,'R_backplate', 10*r_min);
      end

end


%% UNUSED FIELDS
% If some fields of partial geometry were unused they are not removed.
% Throw a warning since they could be mispelled fields causing unexpected
% behaviour (by reverting to default)
if ~isempty(fieldnames(PG))
    wm=strcat('Some of the fields in the partial geometry PG struct are', ...
        ' unused. This could cause unexpected behaviour eg. if some', ...
        ' optional fields were mispelled, leading to a revert to the', ...
        ' default value. The unused fields are shown below.         ');
    warning(wm)
    disp(PG)
end


%% NONINTEGER BLADE WARNINGS
if options.noninteger_blade_warnings
    resid=mod(Z_igv,1)+mod(Z_r_full,1)+mod(Z_r_split,1)+ ...
        +mod(Z_vd,1)+mod(Z_rc,1);
    if resid>1e-10
        warning('Non integer number of blades. Results will be less accurate.')
    end
    if Z_r_split>0&&Z_r_split~=Z_r_full
        warining('Number of splitter and full blades are different.')
    end
end


%% INADEQUATE GEOMETRY WARNINGS
if options.inadequate_geometry_warnings
    if D_1_s/D_2<0.4||D_1_s/D_2>0.75
        warning('D_1_s/D_2 outside the suggested range [0.4,0.75]')
    end
    if D_1_h/D_1_s<0.25||D_1_h/D_1_s>0.75
        warning('D_1_h/D_1_s outside the suggested range [0.25,0.75]')
    end
    if (LR_split>0)&&(LR_split<0.5||LR_split>0.75)
        warning('LR_split outside the suggested range [0.5,0.75]')
    end
    if b_2/D_2<0.03||b_2/D_2>0.09
        warning('b_2/D_2 outside the suggested range [0.03,0.09]')
    end
    if beta_2b<-50||beta_2b>0
        warning('beta_2b outside the suggested range [-50,0]')
    end
    if beta_1b_s<-65||beta_1b_s>-50
        warning('beta_1b_s outside the suggested range [-65,-50]')
    end
    if ceil(Z_r)<10||ceil(Z_r)>30
        warning('Z_r (including splitter fraction if present) outside the suggested range [10,30]')
    end
    if strcmpi(outlet_type,'return channel')
        if R_56<0.8*(b_8-b_6)
            warning('The curvature radius of the crossover bend should be >=0.8*(b_8-b_6)')
        end
    end
end


%% DEGENERATE GEOMETRY ERRORS
if Ra<=0
    error('Roughness must be a positive value')
end
if D_3<1.00*D_2
    error('D_3/D_2 must be >1.00')
end
if has_vaned
    if o_5>pi*D_5*cosd(alpha_5b)/Z_vd
        error('Excessive area ratio in the vaned diffuser,leading to excessive outlet width o5>pi*D_5*cosd(alpha_5b)/Z_vd')
    end
end


%% EXPORT STRUCT
geom.Ra=Ra;
geom.has_igv=has_igv;
geom.D_in_h=D_in_h;
geom.D_in=D_in;
geom.D_in_s=D_in_s;
geom.A_in=A_in;
if has_igv
    geom.alpha_inb=alpha_inb;
    geom.alpha_0b=alpha_0b;
    geom.Z_igv=Z_igv;
    geom.ac_igv=ac_igv;
    geom.c_igv=c_igv;
end
geom.has_inducer=has_inducer;
geom.inducer_type=inducer_type;
geom.L_inducer=L_inducer;
geom.D_0_h=D_0_h;
geom.D_0=D_0;
geom.D_0_s=D_0_s;
geom.A_0=A_0;
if has_inducer
    geom.D_hyd_inducer=D_hyd_inducer;
    geom.inducer_type=inducer_type;
end
geom.is_shrouded=is_shrouded;
geom.D_1_h=D_1_h;
geom.D_1=D_1;
geom.D_1_s=D_1_s;
geom.b_1=b_1;
geom.beta_1b_h=beta_1b_h;
geom.beta_1b=beta_1b;
geom.beta_1b_s=beta_1b_s;
geom.t_b_1=t_b_1;
geom.A_1_eff=A_1_eff;
geom.A_rth=A_rth;
geom.D_2=D_2;
geom.beta_2b=beta_2b;
geom.b_2=b_2;
geom.t_b_2=t_b_2;
geom.A_2_eff=A_2_eff;
geom.Z_r_full=Z_r_full;
geom.Z_r_split=Z_r_split;
geom.LR_split=LR_split;
geom.Z_r=Z_r;
geom.L_z=L_z;
geom.eps_a=eps_a;
geom.eps_r=eps_r;
geom.eps_b=eps_b;
geom.D_3=D_3;
geom.b_3=b_3;
geom.A_3_eff=A_3_eff;
geom.dbdr_vaneless=dbdr_vaneless;
geom.has_vaned=has_vaned;
geom.D_4=D_4;
geom.D_5=D_5;
geom.b_5=b_5;
geom.D_6=D_6;
geom.b_6=b_6;

geom.r_0_h = geom.D_0_h/2;
geom.r_0_s = geom.D_0_s/2;
geom.r_1_h = geom.D_1_h/2;
geom.r_1_s = geom.D_1_s/2;
geom.r_1 = geom.D_1/2;
geom.r_2 = geom.D_2/2;
geom.r_3 = geom.D_3/2;
geom.r_4 = geom.D_4/2;
geom.r_5 = geom.D_5/2;
geom.r_6 = geom.D_6/2;

if has_vaned
    geom.vane_type = vane_type;

    if strcmp(geom.vane_type, 'airfoil')
        geom.b_4=b_4;
        geom.alpha_3b=alpha_3b;
        geom.alpha_5b=alpha_5b;
        geom.t_b_3=t_b_3;
        geom.t_b_5=t_b_5;
        geom.ac_vd =ac_vd;
        geom.alpha_35m=alpha_35m;
        geom.Z_vd=Z_vd;
        geom.AR_vd=AR_vd;
        geom.AS_vd=AS_vd;
        geom.theta_vd=theta_vd;
        geom.L_b_vd=L_b_vd;
        geom.sigma_vd=sigma_vd;
        geom.delta0_vd=delta0_vd;
        geom.o_4=o_4;
        geom.o_5=o_5;
        geom.A_th_4=A_th_4;
        geom.C_r_th_vd=C_r_th_vd;
        geom.A_th_4_eff=A_th_4_eff;
        geom.alpha_th_4=alpha_th_4;
        geom.D_hyd_vd=D_hyd_vd;
        geom.LR_vd=LR_vd;
        geom.DdeltaDinc_vd=DdeltaDinc_vd;
        geom.A_5=A_5;

    elseif strcmp(geom.vane_type, 'wedge')
        geom.b_4=b_4;
        geom.alpha_3b=alpha_3b;
        geom.alpha_5b=alpha_5b;
        geom.t_b_3=t_b_3;
        geom.t_b_5=t_b_5;
        geom.ac_vd =ac_vd;
        geom.alpha_35m=alpha_35m;
        geom.Z_vd=Z_vd;
        geom.AR_vd=AR_vd;
        geom.AS_vd=AS_vd;
        geom.theta_vd=theta_vd;
        geom.L_b_vd=L_b_vd;
        geom.sigma_vd=sigma_vd;
        geom.delta0_vd=delta0_vd;
        geom.o_4=o_4;
        geom.o_5=o_5;
        geom.A_th_4=A_th_4;
        geom.C_r_th_vd=C_r_th_vd;
        geom.A_th_4_eff=A_th_4_eff;
        geom.alpha_th_4=alpha_th_4;
        geom.D_hyd_vd=D_hyd_vd;
        geom.LR_vd=LR_vd;
        geom.DdeltaDinc_vd=DdeltaDinc_vd;
        geom.A_5=A_5;

        geom.wedge_angle = wedge_angle;

    else
        error('Vane type must be "wedge" or "airfoil"')
    end

end

geom.outlet_type=outlet_type;
if strcmpi(outlet_type,'volute')
    geom.volute_type=volute_type;
    geom.D_6=D_6;
    geom.A_volute=A_volute;
    geom.A_exitcone=A_exitcone;
elseif strcmpi(outlet_type,'return channel')
    geom.D_6=D_6;
    geom.R_56=R_56;
    geom.b_6=b_6;
    geom.alpha_6b=alpha_6b;
    geom.t_b_6=t_b_6;
    geom.o_6=o_6;
    geom.alpha_67m=alpha_67m;
    geom.D_7=D_7;
    geom.b_7=b_7;
    geom.alpha_7b=alpha_7b;
    geom.A_7=A_7;
    geom.Z_rc=Z_rc;
    geom.L_b_rc=L_b_rc;
    geom.sigma_rc=sigma_rc;
    geom.delta0_rc=delta0_rc;
    geom.DdeltaDinc_rc=DdeltaDinc_rc;
    geom.D_hyd_rc=D_hyd_rc;
    geom.D_8_h=D_8_h;
    geom.D_8=D_8;
    geom.D_8_s=D_8_s;
    geom.A_8=A_8;
elseif strcmpi(outlet_type,'none')
end

geom.include_backface_geometry=include_backface_geometry;
if include_backface_geometry
%     geom.t_impeller=t_impeller;
%     geom.t_backface=t_backface;
%     geom.D_backface=D_backface;
%     geom.D_shaft=D_shaft;

    geom.backplate_type = backplate_type;
    geom.t_backplate_1 = t_backplate_1;
    geom.t_backplate_2 = t_backplate_2;
    geom.D_backplate_1 = D_backplate_1;
    geom.D_backplate_2 = D_backplate_2;
    geom.r_backplate_1 = r_backplate_1;
    geom.r_backplate_2 = r_backplate_2;
    if strcmp(backplate_type, 'circular_arc')
        geom.R_backplate = R_backplate;
    end
    geom.r_shaft = r_shaft;
    geom.D_shaft = D_shaft;
    geom.L_shaft = L_shaft;
end
%EOF

end
