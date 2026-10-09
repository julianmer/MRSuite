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

} else if (job$task == "about") {
    functions <- ls("package:spant")
    # each function's arguments, a default as the R code that states it ("" for none)
    arguments <- lapply(functions, function(name) {
        fun <- getExportedValue("spant", name)
        if (!is.function(fun)) return(list())
        given <- formals(fun)
        sapply(names(given), function(arg) paste(deparse(given[[arg]]), collapse = " "),
               simplify = FALSE)
    })
    names(arguments) <- functions
    # and its help page, as R prints it, under every name the page documents
    pages <- setNames(list(), character())                   # {} in JSON, even with none
    for (rd in tools::Rd_db("spant")) {
        text <- paste(capture.output(tools::Rd2txt(rd, options = list(underline_titles = FALSE))),
                      collapse = "\n")
        tags <- vapply(rd, function(part) attr(part, "Rd_tag"), "")
        for (alias in unlist(lapply(rd[tags == "\\alias"], as.character))) pages[[alias]] <- text
    }
    out <- list(arguments = arguments, help = pages)

} else if (job$task == "call") {
    # a spant function over each spectrum; arguments that are themselves spectra come
    # per item under job$spectra, everything else is shared
    # spectra come back as files, anything else (an SNR, a table) as a value
    fun <- getExportedValue("spant", job$func)
    values <- rep(list(NA), length(job$inputs))
    for (i in seq_along(job$inputs)) {
        extra <- lapply(job$spectra, function(paths) read_spectrum(paths[[i]], job$refs[[i]]))
        result <- do.call(fun, c(list(read_spectrum(job$inputs[[i]], job$refs[[i]])),
                                 job$args, job$kwargs, extra))
        if (inherits(result, "mrs_data")) {
            write_mrs(result, job$outputs[[i]], format = "nifti", force = TRUE)
        } else if (!is.null(result)) {          # NULL would drop the item from the list
            values[[i]] <- unclass(result)
        }
    }
    out <- list(ok = TRUE, values = values)

} else if (job$task == "fit") {
    basis <- read_basis(job$basis, ref = job$basis_ref)
    opts <- if (length(job$opts)) do.call(abfit_opts, job$opts) else NULL
    rows <- list()
    scales <- numeric()
    for (i in seq_along(job$inputs)) {
        data <- read_spectrum(job$inputs[[i]], job$refs[[i]])
        # ABfit's error estimates depend on the data's scale against the basis' (its
        # CRLB inverts a matrix of both through a pseudo-inverse with a relative
        # tolerance): fit with the data at the basis' scale and put the amplitudes and
        # their sd back afterwards
        resampled <- resample_basis(basis, data)
        scales[i] <- max(Mod(data$data)) / max(Mod(resampled$data))
        fit <- fit_mrs(scale_mrs_amp(data, 1 / scales[i]), resampled,
                       method = job$method, opts = opts, progress = "none", time = FALSE)
        rows[[i]] <- as.list(fit$res_tab[1, ])
    }
    # amp_cols counts the amplitude columns - the basis, then spant's sums such as tCr -
    # which start where the first basis name does
    first <- match(basis$names[1], colnames(fit$res_tab))
    amps <- colnames(fit$res_tab)[first - 1 + seq_len(fit$amp_cols)]
    for (i in seq_along(rows))
        for (col in c(amps, paste0(amps, ".sd")))
            if (is.numeric(rows[[i]][[col]])) rows[[i]][[col]] <- rows[[i]][[col]] * scales[i]
    out <- list(rows = rows, amplitudes = amps)

} else {
    stop("unknown task ", job$task)
}

writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, na = "null"), job$result)
