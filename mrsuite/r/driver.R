####################################################################################################
#                                            driver.R                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-27                                                                              #
#                                                                                                  #
# Purpose: Runs one MRSuite job in R, for the spant bridge: spectra arrive and leave as            #
#          NIfTI-MRS files, everything else as JSON. R runs in a process of its own because        #
#          loading it into Python's alongside NumPy's links two OpenMP runtimes, which aborts.     #
#                                                                                                  #
#          Usage: Rscript driver.R job.json                                                        #
#                                                                                                  #
####################################################################################################

suppressMessages({
    library(spant)
    library(jsonlite)
})

job <- fromJSON(commandArgs(trailingOnly = TRUE)[1], simplifyVector = TRUE,
                simplifyDataFrame = FALSE)

#*************#
#   spectra   #
#*************#
# A spectrum, at the ppm reference the caller states rather than spant's default.
read_spectrum <- function(path, ref) read_mrs(path, format = "nifti", ref = ref)

if (job$task == "info") {
    out <- list(spant = as.character(packageVersion("spant")), r = R.version.string,
                functions = ls("package:spant"))

} else if (job$task == "call") {
    # a spant function over each spectrum; arguments that are themselves spectra come
    # per item under job$spectra, everything else is shared
    fun <- getExportedValue("spant", job$func)
    for (i in seq_along(job$inputs)) {
        extra <- lapply(job$spectra, function(paths) read_spectrum(paths[[i]], job$refs[[i]]))
        result <- do.call(fun, c(list(read_spectrum(job$inputs[[i]], job$refs[[i]])),
                                 job$args, job$kwargs, extra))
        if (!inherits(result, "mrs_data")) stop(job$func, " did not return spectra")
        write_mrs(result, job$outputs[[i]], format = "nifti", force = TRUE)
    }
    out <- list(ok = TRUE)

} else if (job$task == "fit") {
    basis <- read_basis(job$basis, ref = job$basis_ref)
    opts <- if (length(job$opts)) do.call(abfit_opts, job$opts) else NULL
    rows <- list()
    for (i in seq_along(job$inputs)) {
        data <- read_spectrum(job$inputs[[i]], job$refs[[i]])
        fit <- fit_mrs(data, resample_basis(basis, data), method = job$method, opts = opts,
                       progress = "none", time = FALSE)
        rows[[i]] <- as.list(fit$res_tab[1, ])
    }
    # amp_cols counts the amplitude columns - the basis, then spant's sums such as tCr -
    # which start where the first basis name does
    first <- match(basis$names[1], colnames(fit$res_tab))
    out <- list(rows = rows,
                amplitudes = colnames(fit$res_tab)[first - 1 + seq_len(fit$amp_cols)])

} else {
    stop("unknown task ", job$task)
}

writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, na = "null"), job$result)
