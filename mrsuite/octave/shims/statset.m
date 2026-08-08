function opts = statset(varargin)
% STATSET  Octave stand-in for MATLAB's Statistics Toolbox statset.
%
%   opts = statset('nlinfit')                  default options for nlinfit
%   opts = statset(opts, 'Name', value, ...)   override fields on an existing struct
%   opts = statset('Name', value, ...)         defaults with overrides
%
% Only the fields MRS code actually reads are modelled. Unknown names are stored
% rather than rejected, so a caller setting an option this shim ignores still gets
% a usable struct instead of an error.
%
% Part of MRSuite's Octave compatibility shims: Osprey's referencing routines call
% statset/nlinfit, which ship with MATLAB's Statistics and Machine Learning Toolbox
% and have no Octave equivalent.

    opts = struct( ...
        'DerivStep',    eps^(1/3), ...
        'Display',      'off', ...
        'FunValCheck',  'on', ...
        'MaxFunEvals',  200, ...
        'MaxIter',      200, ...
        'OutputFcn',    [], ...
        'Robust',       'off', ...
        'TolFun',       1e-8, ...
        'TolX',         1e-8, ...
        'Tune',         [], ...
        'WgtFun',       'bisquare');

    args = varargin;

    % statset('nlinfit') and friends: a bare function name just means "give me the
    % defaults for that function". We use one default set for all of them.
    if numel(args) == 1 && ischar(args{1})
        return;
    end

    % statset(existingOpts, ...) starts from the struct handed in
    if ~isempty(args) && isstruct(args{1})
        base = args{1};
        fn = fieldnames(base);
        for k = 1:numel(fn)
            opts.(fn{k}) = base.(fn{k});
        end
        args = args(2:end);
    end

    if mod(numel(args), 2) ~= 0
        error('statset: options must be given as name/value pairs.');
    end
    for k = 1:2:numel(args)
        if ~ischar(args{k})
            error('statset: option name must be a string.');
        end
        opts.(args{k}) = args{k + 1};
    end
end
