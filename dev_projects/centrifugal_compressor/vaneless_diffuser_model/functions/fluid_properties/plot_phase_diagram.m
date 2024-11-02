function plot_phase_diagram(prop_x, prop_y, fluid, NameValueArgs)

    arguments
        prop_x (1, 1) string
        prop_y (1, 1) string
        fluid
        NameValueArgs.axes = gca
        NameValueArgs.N_points (1, 1) = 200
        NameValueArgs.plot_saturation_line (1, 1) logical = true
        NameValueArgs.plot_critical_point (1, 1) logical = true
        NameValueArgs.plot_triple_point (1, 1) logical = false
        NameValueArgs.plot_spinodal_line (1, 1) logical = false
        NameValueArgs.spinodal_line_method (1, 1) string = 'standard'
        NameValueArgs.spinodal_line_color = 0.5*[1, 1, 1]
        NameValueArgs.spinodal_line_width = 0.75
        NameValueArgs.plot_quality_isolines (1, 1) logical  = false
        NameValueArgs.plot_pseudocritical_line (1, 1) logical = false
        NameValueArgs.quality_levels (:, 1) double = 0.0:0.1:0.9
        NameValueArgs.quality_labels (1, 1) logical = false
        NameValueArgs.show_in_legend (1, 1) logical = false
    end
    
    if NameValueArgs.show_in_legend
        visible = 'on';
    else
        visible = 'off';
    end
    
    % Plot saturation line
    if NameValueArgs.plot_saturation_line
        [sat_liq, sat_vap] = compute_saturation_line(fluid, N_points=NameValueArgs.N_points);
        plot(NameValueArgs.axes, [sat_liq.(prop_x), sat_vap.(prop_x)], [sat_liq.(prop_y), sat_vap.(prop_y)], 'k', LineWidth=1.25, DisplayName='Saturation line', HandleVisibility=visible)
    end
    
    % Plot spinodal line
    if NameValueArgs.plot_spinodal_line
        [spinodal_liq, spinodal_vap] = compute_spinodal_line(fluid, N_points=NameValueArgs.N_points, method=NameValueArgs.spinodal_line_method);
        plot(NameValueArgs.axes, [spinodal_liq.(prop_x), spinodal_vap.(prop_x)], [spinodal_liq.(prop_y), spinodal_vap.(prop_y)], Color=NameValueArgs.spinodal_line_color, LineWidth=NameValueArgs.spinodal_line_width, DisplayName='Spinodal line', HandleVisibility=visible)
    end
    
    % Compute vapor quality isocurves
    if NameValueArgs.plot_quality_isolines
        t1 = logspace(log10(1-0.9999), log10(0.1), ceil(NameValueArgs.N_points/2));
        t2 = logspace(log10(0.1), log10(1-(fluid.Ttriple)/fluid.T_critical), floor(NameValueArgs.N_points/2));
        T_quality = (1-[t1 t2])*fluid.T_critical;
        Q_quality = NameValueArgs.quality_levels;
        x_quality = zeros(numel(Q_quality), numel(T_quality));
        y_quality = zeros(numel(Q_quality), numel(T_quality));     
        for i = 1:numel(Q_quality)
            for j = 1:numel(T_quality)
                fluid.update(py.CoolProp.QT_INPUTS, Q_quality(i), T_quality(j))
                x_quality(i, j) = fluid.(prop_x);
                y_quality(i, j) = fluid.(prop_y);
            end
        end
        [C, h] = contour(NameValueArgs.axes, x_quality, y_quality, repmat(Q_quality, 1, numel(T_quality)), Q_quality, Color="black", LineStyle=':', LineWidth=0.75, DisplayName='Quality isolines', HandleVisibility=visible);
        if NameValueArgs.quality_labels
            clabel(C,h, Interpreter="Latex", LabelSpacing=500, FontSize=9)
        end
    
    end
    
    % Plot pseudocritical line
    if NameValueArgs.plot_pseudocritical_line
      pseudo_critical_line = compute_pseudocritical_line(fluid);
      plot(NameValueArgs.axes, pseudo_critical_line.(prop_x), pseudo_critical_line.(prop_y), Color="black", LineStyle='-', LineWidth=0.25, DisplayName='Pseudocritical line', HandleVisibility=visible)
    end
    
    % Plot critical point
    if NameValueArgs.plot_critical_point
        fluid.update(py.CoolProp.DmassT_INPUTS, fluid.rhomass_critical, fluid.T_critical)
        plot(NameValueArgs.axes, fluid.(prop_x), fluid.(prop_y), 'ko', MarkerSize=4.5, MarkerFaceColor='w', DisplayName='Critical point', HandleVisibility='off')
    end
    
    % Plot triple point
    if NameValueArgs.plot_triple_point
        fluid.update(py.CoolProp.QT_INPUTS, 0.00, fluid.Ttriple)
        plot(NameValueArgs.axes, fluid.(prop_x), fluid.(prop_y), 'ko', MarkerSize=4.5, MarkerFaceColor='w', DisplayName='Triple point liquid', HandleVisibility='off')
        fluid.update(py.CoolProp.QT_INPUTS, 1.00, fluid.Ttriple)
        plot(NameValueArgs.axes, fluid.(prop_x), fluid.(prop_y), 'ko', MarkerSize=4.5, MarkerFaceColor='w', DisplayName='Triple point vapor', HandleVisibility='off')
    end

end
