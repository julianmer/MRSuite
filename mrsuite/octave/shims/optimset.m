function opts = optimset(varargin)
% OPTIMSET  Octave stand-in for MATLAB's optimset, for the optimisers MRSuite shims.
%
%   opts = optimset('lsqnonlin')            defaults for a named solver
%   opts = optimset(opts, 'Name', val, ...) override fields
%
% Octave ships an optimset of its own, but it validates option names against its own
% solvers and rejects MATLAB ones such as 'Algorithm'. Osprey calls
% optimset(options,'Display','off','FunValCheck','on','Algorithm','levenberg-marquardt'),
% so this accepts any name rather than policing the list.

    opts = struct('Algorithm', 'levenberg-marquardt', ...
                  'Display', 'off', ...
                  'FunValCheck', 'on', ...
                  'Jacobian', 'off', ...
                  'MaxFunEvals', 1e4, ...
                  'MaxIter', 400, ...
                  'TolFun', 1e-8, ...
                  'TolX', 1e-8, ...
                  'FinDiffRelStep', eps^(1/3));

    args = varargin;
    if numel(args) == 1 && (ischar(args{1}) || isa(args{1}, 'function_handle'))
        return;                                   % optimset('lsqnonlin')
    end
    if ~isempty(args) && isstruct(args{1})
        base = args{1};
        for f = fieldnames(base)'
            opts.(f{1}) = base.(f{1});
        end
        args = args(2:end);
    end
    if mod(numel(args), 2) ~= 0
        error('optimset: options must be name/value pairs.');
    end
    for k = 1:2:numel(args)
        opts.(args{k}) = args{k + 1};
    end
end
