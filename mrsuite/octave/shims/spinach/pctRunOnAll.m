function pctRunOnAll(cmd)
% PCTRUNONALL  Spinach-scoped serial stand-in: runs the command in the client.
    evalin('base', cmd);
end
