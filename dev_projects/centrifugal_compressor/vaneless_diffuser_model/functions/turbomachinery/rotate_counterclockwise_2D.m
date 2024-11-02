function [x, y] = rotate_counterclockwise_2D(x, y, angle)
    
    % Define rotation metrix
    R = [+cos(angle), -sin(angle);
         +sin(angle), +cos(angle)];
    
    % Rotate the coordinates
    coords = R*[x; y];

    % Rename output variables
    x = coords(1,:);
    y = coords(2,:);
    
end
