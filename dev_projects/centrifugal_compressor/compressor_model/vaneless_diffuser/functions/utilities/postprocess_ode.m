function ode_out = postprocess_ode(t, y, ode_handle)
    % POSTPROCESS_ODE Extract and store additional properties from an ODE solution.
    %
    % Syntax:
    %   ode_out = postprocess_ode(t, y, ode_handle)
    %
    % Description:
    %   The function POSTPROCESS_ODE iterates through the points of an ODE solution 
    %   and retrieves additional properties or variables using the provided ODE handle.
    %   It assumes that the ODE handle returns, as an additional output, a structure 
    %   containing the desired properties to be exported.
    %
    % Input Arguments:
    %   - t          : Vector containing the time (or independent variable) points.
    %   - y          : Matrix where each row represents the state values at 
    %                  corresponding points in 't'.
    %   - ode_handle : Handle to the ODE function. It is expected that this function 
    %                  has the signature [dy, props] = ode_handle(t, y) where 
    %                  'dy' is the ODE's derivative and 'props' is a structure 
    %                  containing the additional properties.
    %
    % Output Argument:
    %   - ode_out    : A structure where each field represents a property retrieved 
    %                  from the ODE handle. The fields contain arrays with values 
    %                  corresponding to each point in 't'.
    %
    % Example:
    %               
    %     % Parameters
    %     k = 2.0; % Spring constant
    %     m = 1.0; % Mass
    %     c = 0.1; % Damping coefficient
    %     
    %     % Solve the ODE
    %     t = linspace(0, 10, 200);
    %     [T, Y] = ode45(@(t,y) damped_harmonic_oscillator(t, y, k, m, c), t, [1 0]);
    %     ode_results = postprocess_ode(T, Y, @(t,y) damped_harmonic_oscillator(t, y, k, m, c));
    %     
    %     % Plotting
    %     figure; hold on; box on; grid off
    %     title('Energy distribution in a damped harmonic oscillatior');
    %     xlabel('Time (s)');
    %     ylabel('Energy (J)');
    %     plot(T, ode_results.potentialEnergy, DisplayName="Potential energy");
    %     plot(T, ode_results.kineticEnergy, DisplayName="Kinetic energy" );
    %     plot(T, ode_results.totalEnergy, DisplayName="Total energy" );
    %     legend(Location="northeast");
    %     
    %     % Define the ODE and properties
    %     function [dy, props] = damped_harmonic_oscillator(t, y, k, m, c)
    %         % Extract the state variables
    %         x = y(1);
    %         v = y(2);
    %     
    %         % ODE system derivatives
    %         dx = v;
    %         dv = (-k * x - c * v) / m;
    %         dy = [dx; dv];
    %     
    %         % Compute additional properties
    %         props = struct();
    %         props.potentialEnergy = 0.5 * k * x^2;
    %         props.kineticEnergy = 0.5 * m * v^2;
    %         props.totalEnergy = props.potentialEnergy + props.kineticEnergy;
    %         
    %     end
    %
    %
    % Note:
    %   Ensure that the ODE handle is implemented in a manner that provides the 
    %   additional properties as a structure in its second output.
    %
    % See also: ode45, ode23, fieldnames
    %
    % Author:
    %   Roberto Agromayor
    %   roagr@dtu.dk
    %   Date: 19.09.2023
    

    % Loop over all the points of the ODE solution
    ode_out = struct();
    for i = 1:numel(t)

        % Retrieve structure storing additional variables from the ODE
        [~, ode_out_current] = ode_handle(t(i), y(i,:));

        % Store properties in an struct of arrays
        if i == 1
            
            % Initialize structure for the first point
            fields = fieldnames(ode_out_current);
            for f = 1:numel(fields)
                ode_out.(fields{f}) = [ode_out_current.(fields{f})];
            end
    
        else

            % Append properties to structure for subsequent points
            for f = 1:numel(fields)
                ode_out.(fields{f}) = [ode_out.(fields{f}); ode_out_current.(fields{f})];
            end

        end
    end

end
