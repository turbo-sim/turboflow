classdef FluidProperties < handle

    % Class developed by Roberto Agromayor to compute a grid of
    % thermodynamic properties using CoolProp
    
    properties

        fluid_name

        p_min
        p_max
        T_min
        T_max
        d_min
        d_max

        p_triple
        T_triple
        d_triple
        h_triple
        s_triple

        p_critical
        T_critical
        d_critical
        h_critical
        s_critical

        n_sat
        p_sat
        T_sat
        d_liq
        d_vap
        h_liq
        h_vap
        s_liq
        s_vap

        n_grid_T
        n_grid_d
        p_grid
        T_grid
        d_grid
        h_grid
        s_grid
        Z_grid
        kT_grid
        kP_grid
%         Q_grid

    end


    methods
        
        % Constructor method
        function obj = FluidProperties(fluid, nameValueArgs)

            arguments
                fluid (1, 1) string
                nameValueArgs.n_sat (1, 1) double = 200
                nameValueArgs.n_grid_T (1, 1) double = 50
                nameValueArgs.n_grid_d (1, 1) double = 50
                nameValueArgs.computeProperties (1, 1) logical = true
            end
            
            % Define the fluid
            obj.fluid_name = fluid;

            % Update the number of points of the grid
            obj.n_sat = nameValueArgs.n_sat;
            obj.n_grid_T = nameValueArgs.n_grid_T;
            obj.n_grid_d = nameValueArgs.n_grid_d;

            % Evaluate non-trivial thermodynamic properties
            if nameValueArgs.computeProperties

                % Compute thermodynamic properties along saturation line 
                obj.compute_saturation_properties();
    
                % Compute thermodynamic properties on temperature-density grid
                obj.compute_property_grids();

            end

        end


        % Define the fluid of the class
        function set.fluid_name(obj, fluid)
            
            % Update fluid name
            obj.fluid_name = fluid;
            
            % Update the thermodynamic properties of the fluid
            obj.compute_critical_point_properties();
            obj.compute_triple_point_properties();
            obj.set_grid_limits(updateGrid=false);

            % Delete previous thermodynamic grids (but not the points)
            fields = fieldnames(obj);
            fields = setdiff(fields, {'n_sat', 'n_grid_T', 'n_grid_d'});
            for i = 1:numel(fields)
                if contains(fields{i}, {'grid', 'sat', 'liq', 'vap'})
                    obj.(fields{i}) = [];
                end
            end

            fprintf("Fluid set to %s\n", fluid)

        end


        % Compute thermodynamic properties for the current fluid
        function obj = compute_thermodynamic_properties(obj, nameValueArgs)

            arguments
                obj
                nameValueArgs.n_sat (1, 1) double = obj.n_sat
                nameValueArgs.n_grid_T (1, 1) double = obj.n_grid_T
                nameValueArgs.n_grid_d (1, 1) double = obj.n_grid_d
            end

            % Start message
            fprintf("Computing properties for %s\n", obj.fluid_name)

            % Update the number of points along saturation lines
            obj.n_sat = nameValueArgs.n_sat;

            % Update the number of points of the grid
            obj.n_grid_T = nameValueArgs.n_grid_T;
            obj.n_grid_d = nameValueArgs.n_grid_d;
            
            % Compute thermodynamic properties along saturation line 
            obj.compute_saturation_properties();

            % Compute thermodynamic properties on temperature-density grid
            obj.compute_property_grids();

        end


        % Define the limits of the thermodynamic region
        function obj = set_grid_limits(obj, nameValueArgs)

            arguments
                obj
                nameValueArgs.p_min (1, 1) double = obj.p_triple
                nameValueArgs.p_max (1, 1) double = 10.0*obj.p_critical
                nameValueArgs.T_min (1, 1) double = 1.0*obj.T_triple
                nameValueArgs.T_max (1, 1) double = 3.0*obj.T_critical
                nameValueArgs.updateGrid logical = false
            end
            
            % Define pressure-temperature limits
            obj.p_min = nameValueArgs.p_min;
            obj.p_max = nameValueArgs.p_max;
            obj.T_min = nameValueArgs.T_min;
            obj.T_max = nameValueArgs.T_max;

            % Compute corresponding density limits
            obj.d_min = PropsSI_array('D', 'T', obj.T_max, 'P', obj.p_min, obj.fluid_name);
            obj.d_max = PropsSI_array('D', 'T', obj.T_min, 'P', obj.p_max, obj.fluid_name);
            if nameValueArgs.updateGrid
               obj.compute_property_grids();
            end

        end

        
        % Get properties at the critical point
        function obj = compute_critical_point_properties(obj)
            obj.p_critical = PropsSI_array("P_CRITICAL", obj.fluid_name);
            obj.T_critical = PropsSI_array("T_CRITICAL", obj.fluid_name);
            obj.d_critical = PropsSI_array("D", "T", obj.T_critical, "P", obj.p_critical, obj.fluid_name);
            obj.s_critical = PropsSI_array("S", "T", obj.T_critical, "P", obj.p_critical, obj.fluid_name);
            obj.h_critical = PropsSI_array("H", "T", obj.T_critical, "P", obj.p_critical, obj.fluid_name);
        end


        % Define properties at the triple point
        function obj = compute_triple_point_properties(obj)
            obj.p_triple = PropsSI_array("P_TRIPLE", obj.fluid_name);
            obj.T_triple = PropsSI_array("T_TRIPLE", obj.fluid_name);
            obj.d_triple = PropsSI_array("D", "T", obj.T_triple, "P", obj.p_triple+100, obj.fluid_name);  % Move 100 Pa away from triple point
            obj.h_triple = PropsSI_array("H", "T", obj.T_triple, "P", obj.p_triple+100, obj.fluid_name);  % Move 100 Pa away from triple point
            obj.s_triple = PropsSI_array("S", "T", obj.T_triple, "P", obj.p_triple+100, obj.fluid_name);  % Move 100 Pa away from triple point
        end


        % Calculate properties along the saturation line
        function obj = compute_saturation_properties(obj, nameValueArgs)

            arguments
                obj
                nameValueArgs.n_sat (1, 1) double = obj.n_sat
            end
            
            fprintf('Computing saturation lines of size %d\n', obj.n_sat)
            obj.n_sat = nameValueArgs.n_sat;       
            obj.T_sat = linspace(obj.T_triple, obj.T_critical, obj.n_sat)';
            obj.p_sat = PropsSI_array("P", "T", obj.T_sat, "Q", 0.00, obj.fluid_name);
            % p_sat = linspace(p_triple, p_critical, 200)';
            % T_sat = PropsSI_array("T", "P", p_sat, "Q", 0.00, obj.fluid);
            obj.d_liq = PropsSI_array("D", "T", obj.T_sat, "Q", 0.00, obj.fluid_name);
            obj.d_vap = PropsSI_array("D", "T", obj.T_sat, "Q", 1.00, obj.fluid_name);
            obj.s_liq = PropsSI_array("S", "T", obj.T_sat, "Q", 0.00, obj.fluid_name);
            obj.s_vap = PropsSI_array("S", "T", obj.T_sat, "Q", 1.00, obj.fluid_name);
            obj.h_liq = PropsSI_array("H", "T", obj.T_sat, "Q", 0.00, obj.fluid_name);
            obj.h_vap = PropsSI_array("H", "T", obj.T_sat, "Q", 1.00, obj.fluid_name);
            fprintf('Saturation line computations complete\n')

        end


        % Calculate properties on a temperature density grid
        function obj = compute_property_grids(obj, nameValueArgs)

            arguments
                obj
                nameValueArgs.n_grid_T (1, 1) double = obj.n_grid_T
                nameValueArgs.n_grid_d (1, 1) double = obj.n_grid_d
            end

            % Update the number of points stored in the class
            obj.n_grid_T = nameValueArgs.n_grid_T;
            obj.n_grid_d = nameValueArgs.n_grid_d;

            % Define the temperature vector (linear spacing between the
            % minimum temperature and the maximum temperature)
            T_vector = linspace(obj.T_min, obj.T_max, obj.n_grid_T);

            % Define the density vector (logarithmic spacing up to the
            % critical point and linear spacing after the critical point)
            d_vector_subcritical_1 = logspace(log10(obj.d_min), log10(obj.d_critical)/2, floor(obj.n_grid_d/2));
            d_vector_subcritical_2 = logspace(log10(obj.d_critical)/2, log10(obj.d_critical), ceil(obj.n_grid_d/2));
            d_vector_subcritical = [d_vector_subcritical_1(1:end-1), d_vector_subcritical_2];
            d_vector_supercritical = linspace(obj.d_critical, obj.d_max, ceil(obj.n_grid_d/2)+1);
            d_vector = [d_vector_subcritical(1:end-1), d_vector_supercritical];

            % Define the temperature-density grid
            [obj.T_grid, obj.d_grid] = meshgrid(T_vector, d_vector);

            % Compute the thermodynamic properties on the grid
            fprintf('Computing property grids of size %dx%d\n', obj.n_grid_d, obj.n_grid_T)
            obj.p_grid = PropsSI_array("P", "T", obj.T_grid, "D", obj.d_grid, obj.fluid_name);
            obj.h_grid = PropsSI_array("H", "T", obj.T_grid, "D", obj.d_grid, obj.fluid_name);
            obj.s_grid = PropsSI_array("S", "T", obj.T_grid, "D", obj.d_grid, obj.fluid_name);
            obj.Z_grid = PropsSI_array("Z_def", "T", obj.T_grid, "D", obj.d_grid, obj.fluid_name);
            obj.kT_grid = PropsSI_array("isobaric_expansion_2phase", "T", obj.T_grid, "D", obj.d_grid, obj.fluid_name);
            obj.kP_grid = PropsSI_array("isothermal_compressibility_2phase", "T", obj.T_grid, "D", obj.d_grid, obj.fluid_name);
            fprintf('Property grid calculations complete\n')

        end


        function obj = update_grid_size(obj, nameValueArgs)

            arguments
                obj
                nameValueArgs.n_grid_T (1, 1) double = obj.n_grid_T
                nameValueArgs.n_grid_d (1, 1) double = obj.n_grid_d
                nameValueArgs.updateGrid logical = true
            end
            
            % Update the number of points of the grid
            obj.n_grid_T = nameValueArgs.n_grid_T;
            obj.n_grid_d = nameValueArgs.n_grid_d;
            fprintf('Grid size updated to %dx%d\n', obj.n_grid_d, obj.n_grid_T)

            % Update the thermodynamic properties
            if nameValueArgs.updateGrid
               obj.compute_property_grids();
            end

        end


    end

end



