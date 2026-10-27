function save(varargin)
% SAVE  MRS4Brain-scoped wrapper of Octave's save.
% MATLAB's save writes a MAT file by default and stores objects (classdef
% instances, datetime) with their class. Octave defaults to its text format and
% cannot save classdef objects: here the default format is MAT v7 binary, and
% named variables holding objects (also inside structs and cells) are stored as
% structs of their public properties (they load back as structs). Format flags
% given by the caller are kept; '-struct' calls pass through.
    isopt = cellfun(@(a) ischar(a) && ~isempty(a) && a(1) == '-', varargin);
    formats = {'-text', '-binary', '-mat', '-mat-binary', '-v7', '-7', '-mat7-binary', ...
               '-v6', '-6', '-v4', '-4', '-hdf5', '-float-binary', '-float-hdf5', '-zip', '-z'};
    args = varargin;
    args(strcmpi(args, '-v7.3')) = {'-mat7-binary'};
    if ~any(ismember(lower(args(isopt)), formats)), args = [{'-mat7-binary'} args]; isopt = [true isopt]; end
    if any(strcmp(args, '-struct')) || any(strcmp(args, '-'))
        builtin('save', args{:}); return
    end
    rest = args(~isopt); opts = args(isopt);
    file = rest{1}; names = rest(2:end);
    if isempty(names), names = evalin('caller', 'who'); end
    s = struct();
    for k = 1:numel(names), s.(names{k}) = plainify(evalin('caller', names{k})); end
    builtin('save', opts{:}, file, '-struct', 's');
end

function v = plainify(v)
    if isobject(v) && ~isa(v, 'containers.Map')
        w = warning('off', 'all'); v = struct(v); warning(w);
    end
    if isstruct(v)
        f = fieldnames(v);
        for i = 1:numel(v)
            for j = 1:numel(f), v(i).(f{j}) = plainify(v(i).(f{j})); end
        end
    elseif iscell(v)
        v = cellfun(@plainify, v, 'UniformOutput', false);
    end
end
