function out = mrsuite_oxsa_fit(fid, dwell, frequency, begin_time, prior, known, options)
% MRSUITE_OXSA_FIT  One spectrum through OXSA's AMARES, quietly, as one struct.
%
%   out = mrsuite_oxsa_fit(fid, dwell, frequency, begin_time, prior, known, options)
%
% FID is the complex time-domain signal (a column), DWELL its dwell time in s, FREQUENCY
% the spectrometer frequency in MHz, BEGIN_TIME the time of the first point in s (OXSA's
% beginTime, for its first-order phase). PRIOR is the name of a set in
% AMARES.priorKnowledge, or a struct of OXSA's three tables (bounds, initialValues,
% priorKnowledge), each a struct array or, as oct2py hands a list over, a cell of structs.
% KNOWN is the cell of set names MRSuite offers (its oxsa.priors(): the folder listed once,
% there; OXSA's own listAllPkFiles globs 'pk*.m', which matches nothing where file names are
% case-sensitive). OPTIONS is a struct of amaresFit's name/value options (MaxIter, TolFun, ...).
%
% OXSA prints the tables of a prior as it assembles them, and amaresFit reports unless told
% to be quiet; both are kept off the session's output here. The peak names come back
% flattened, one per fitted component, as amaresFit's results are laid out.
%
% MRSuite's driver code for OXSA, on the path under every runtime.

    if ischar(prior)
        if ~any(strcmp(prior, known))
            error('OXSA has no prior knowledge ''%s''; it holds: %s', prior, strjoin(known, ', '));
        end
        evalc(sprintf('pk = AMARES.priorKnowledge.%s();', prior));
    else
        pk = prior;
        tables = {'bounds', 'initialValues', 'priorKnowledge'};
        for k = 1:numel(tables)
            if iscell(pk.(tables{k}))
                pk.(tables{k}) = [pk.(tables{k}){:}];
            end
        end
        evalc('AMARES.checkConstraints(pk);');
    end

    n = numel(fid);
    expt.samples = n;
    expt.imagingFrequency = frequency;
    expt.dwellTime = dwell;
    expt.beginTime = begin_time;
    expt.timeAxis = (0:n-1).' * dwell + begin_time;
    expt.ppmAxis = (((0:n-1) - floor(n / 2)) / (n * dwell) / frequency).';

    args = {};
    names = fieldnames(options);
    for k = 1:numel(names)
        args(end+1:end+2) = {names{k}, options.(names{k})};
    end
    [results, status, ~, crb] = AMARES.amaresFit(double(fid(:)), expt, pk, 0, ...
                                                  'quiet', true, args{:});

    peaks = {};
    for k = 1:numel(pk.initialValues)
        name = pk.initialValues(k).peakName;
        if iscell(name)
            peaks = [peaks, name];
        else
            peaks{end+1} = name;
        end
    end
    out.names = peaks;
    out.results = results;
    out.crb = crb;
    out.reference = AMARES.getActualRefPeakDx(pk);
    out.exitflag = status.EXITFLAG;
    out.iterations = status.OUTPUT.iterations;
    out.relative_norm = status.relativeNorm;
    out.residual_norm = status.resNormSq;
end
