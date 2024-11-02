function [output,S] = getfield_else_default(S,fieldname,default)
if isfield(S,fieldname)
    output=S.(fieldname);
    S=rmfield(S,fieldname);
elseif nargin==2
    error('%s is a mandatory input',fieldname)
else
    output=default;
end

end

