function g = gray_variant(m)
%GRAY   Linear gray-scale color map
%   GRAY(M) returns an M-by-3 matrix containing a gray-scale colormap.
%   GRAY, by itself, is the same length as the current figure's
%   colormap. If no figure exists, MATLAB uses the length of the
%   default colormap.
%
%   For example, to reset the colormap of the current figure:
%
%             colormap(gray)
%
%   See also HSV, HOT, COOL, BONE, COPPER, PINK, FLAG, 
%   COLORMAP, RGBPLOT.

%   Copyright 1984-2015 The MathWorks, Inc.

if nargin < 1
   f = get(groot,'CurrentFigure');
   if isempty(f)
      m = size(get(groot,'DefaultFigureColormap'),1);
   else
      m = size(f.Colormap,1);
   end
end

% Lighter grey colormap
factor_1 = 0.50;
factor_2 = 1.00;
g = linspace(factor_1, factor_2, m)';
g = [g g g];
