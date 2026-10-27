function MRS_struct = GannetPreInitialiseGUIVersion(config_path, MRS_struct)
% GANNETPREINITIALISEGUIVERSION  Gannet's settings, from a JSON file of MRSuite's.
%
%   MRS_struct = GannetPreInitialiseGUIVersion(config_path, MRS_struct)
%
% GannetLoad(true, config_path, ...) reads its settings through this function instead of
% GannetPreInitialise.m, which a user edits by hand; Gannet's GUI provides it, and the
% published toolbox does not. MRSuite provides it so that a call can choose the settings
% without editing Gannet: Gannet's own GannetPreInitialise sets every default first, then
% each field of the JSON file CONFIG_PATH replaces MRS_struct.p.<field> (HERMES, target,
% vox, ...). JSON arrays decode as columns, Gannet's lists are rows, so they are turned.
%
% MRSuite's driver code for Gannet, on the path under every runtime.

    MRS_struct = GannetPreInitialise(MRS_struct);
    cfg = jsondecode(fileread(config_path));
    f = fieldnames(cfg);
    for k = 1:numel(f)
        v = cfg.(f{k});
        if any(strcmp(f{k}, {'target', 'vox'})) && ischar(v), v = {v}; end
        if iscell(v), v = reshape(v, 1, []); end
        MRS_struct.p.(f{k}) = v;
    end
end
