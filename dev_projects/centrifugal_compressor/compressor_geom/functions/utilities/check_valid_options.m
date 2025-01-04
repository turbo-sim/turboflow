function check_valid_options(field_value, valid_options)
    
    if ~any(strcmp(field_value, valid_options))

        % Write list of the valid options
        options = "";
        for i = 1:numel(valid_options)
            options = strcat(options, "\t- ", valid_options(i), "\n");
        end

        % Raise error
        error("%s is '%s', but the valid options are:\n%s", inputname(1), field_value, sprintf(options))

    end

end