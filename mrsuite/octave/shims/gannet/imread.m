function varargout = imread(varargin)
% IMREAD  Gannet-scoped wrapper of Octave's imread.
% MATLAB's PNG option 'BackgroundColor','none' returns the image without
% compositing it over a background colour. Octave's imread never composites, so
% the option is dropped and the call passed on unchanged.
    orig = mrsuite_original('imread', fileparts(mfilename('fullpath')));
    args = varargin;
    k = find(cellfun(@(a) ischar(a) && strcmpi(a, 'BackgroundColor'), args), 1);
    if ~isempty(k) && k < numel(args) && ischar(args{k + 1}) && strcmpi(args{k + 1}, 'none')
        args(k:k + 1) = [];
    end
    [varargout{1:max(nargout, 1)}] = orig(args{:});
end
