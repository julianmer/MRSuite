function tf = isMATLABReleaseOlderThan(release, varargin)
% ISMATLABRELEASEOLDERTHAN  Octave stand-in for MATLAB's (R2020b), against an emulated release.
%
%   tf = isMATLABReleaseOlderThan('R2025a')
%
% MATLAB: true when the running release is older than RELEASE. Octave has no MATLAB
% release, so it answers for the one mrsuite_matlab_release names: R2024b by default,
% another where a toolbox's shim folder names the release whose functions it supplies.
% The release-type and update arguments are ignored.
%
% Part of MRSuite's Octave compatibility shims: Gannet and Spinach branch on it.

    tf = release_number(mrsuite_matlab_release()) < release_number(release);
end

function v = release_number(r)
    t = regexp(char(r), '^R(\d{4})([ab])', 'tokens', 'once');
    v = str2double(t{1}) + 0.5 * (t{2} == 'b');
end
