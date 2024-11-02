function value_out = PropsSI_array(varargin)
    % Wrapper around CoolProp high-level interface
    % This function accepts arrays of any shape as input variables

    % Calculate trivial thermodynamic properties
    if nargin == 2
        
        % Call Coolprop wrapper
        value_out = PropsSI_scalar(varargin{:});

    % Calculate non-trivial thermodynamic properties
    elseif nargin == 6
        
        % Rename arguments
        name_out = varargin{1};
        name_1 = varargin{2};
        value_1 = varargin{3};
        name_2 = varargin{4};
        value_2 = varargin{5};
        fluid_name = varargin{6};

        % Check the shape of the arguments
        if numel(value_1) == 1 || numel(value_2) == 1 || isequal(size(value_1), size(value_2))

            % Broadcast inputs implicitly
            value_1 = value_1 + 0*value_2;
            value_2 = value_2 + 0*value_1;
    
            % Element-wise function call to CoolProp
            value_out = arrayfun(@(x, y) PropsSI_scalar(name_out, name_1, x, name_2, y, fluid_name), value_1, value_2);

        else 
            
            % The arguments do not have the right shapes
            errID = 'thermodynamicEvaluation:inputError';
            msgtext = 'The input property values should be: \n\t(1) scalar and scalar \n\t(2) array and scalar \n\t(3) scalar and array \n\t(4) arrays of the same size';
            throw(MException(errID,msgtext));

        end

    % Invalid number of arguments
    else
        
        errID = 'thermodynamicEvaluation:inputError';
        msgtext = 'The number of arguments must be 2 or 6';
        throw(MException(errID,msgtext));

    end


end


