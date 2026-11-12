####################################################################################################
#                                            specvis.R                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 10/11/26                                                                                #
#                                                                                                  #
# Purpose: Runs one MRSuite job in R, for the SpecVis bridge: its functions sourced from its       #
#          checkout, which is the working directory as they expect. Tables arrive and leave as    #
#          CSV files, a figure leaves as SpecVis saves it, everything else goes as JSON.           #
#                                                                                                  #
#          Usage: Rscript specvis.R job.json                                                       #
#                                                                                                  #
####################################################################################################

suppressPackageStartupMessages(library(jsonlite))

job <- fromJSON(commandArgs(trailingOnly = TRUE)[1], simplifyVector = TRUE,
                simplifyDataFrame = FALSE)
setwd(job$checkout)

#***********#
#   shims   #
#***********#
# In front of the functions' own calls, on Linux and on today's R: a package missing is said,
# not installed into the user's library; a function naming a file by the wrong case
# (summarySE.r) still finds it; wilcox.test's formula method takes `paired` as R < 4.4 did.
install.packages <- function(pkgs, ...)
    stop("SpecVis needs the R packages ", paste(pkgs, collapse = ", "),
         ": install them, or use the Docker runtime", call. = FALSE)
source <- function(file, ...)
    base::source(if (file.exists(file)) file else sub("\\.r$", ".R", file), ...)
wilcox.test <- function(x, ..., paired = FALSE) {
    if (!inherits(x, "formula")) return(stats::wilcox.test(x, ..., paired = paired))
    frame <- model.frame(x)
    groups <- split(frame[[1]], factor(frame[[2]]))
    if (length(groups) != 2) stop("grouping factor must have exactly 2 levels")
    stats::wilcox.test(groups[[1]], groups[[2]], ..., paired = paired)
}

suppressPackageStartupMessages(source("functions/dependencies.R"))
functions <- list.files("functions", pattern = "^spvs_.*\\.R$")
for (file in functions) source(file.path("functions", file))

#************#
#   tables   #
#************#
read_table <- function(path) read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)

# a list column (a post hoc table's methods) as text, which CSV can carry
flat <- function(table) {
    table[] <- lapply(table, function(column)
        if (is.list(column)) vapply(column, function(v) paste(v, collapse = ", "), "") else column)
    table
}

if (job$task == "info") {
    out <- list(r = R.version.string, ggplot2 = as.character(packageVersion("ggplot2")),
                functions = sub("\\.R$", "", functions))

} else if (job$task == "load") {
    # Osprey's csv gives one table; LCModel's and TARQUIN's folders the amplitudes, CRLBs and
    # diagnostics as a list
    data <- spvs_importResults(job$path, job$quant)
    tables <- if (is.data.frame(data)) list(amplitudes = data) else
        list(amplitudes = as.data.frame(data[[1]]), crlbs = as.data.frame(data[[2]]),
             diagnostics = as.data.frame(data[[3]]))
    for (name in names(tables))
        write.csv(tables[[name]], file.path(job$folder, paste0(name, ".csv")), row.names = FALSE)
    out <- list(tables = as.list(names(tables)))                    # a list: never unboxed

} else if (job$task == "plot") {
    args <- job$args
    if (!is.null(job[["frame"]])) args$dataFrame <- read_table(job[["frame"]])   # exact: $ would
    if (!is.null(job[["frames"]])) args$dataFrame <- lapply(job[["frames"]], read_table)  # match both
    for (name in job$as_list) args[[name]] <- as.list(args[[name]])  # measures the raincloud takes
    # spvs_BlandAltman reads its limits from these two, not from its arguments
    lowerLimit <- args$lowerLimits
    upperLimit <- args$upperLimits
    made <- do.call(get(job$func), args)
    plot <- if (inherits(made, "ggplot")) made else made[[1]]       # spvs_Violin's table beside it
    for (path in job$outputs)
        ggsave(path, plot, width = job$size[1], height = job$size[2], dpi = job$dpi,
               device = if (grepl("\\.pdf$", path)) cairo_pdf else NULL)
    out <- list(outputs = job$outputs)

} else if (job$task == "statistics") {
    stats <- spvs_Statistics(read_table(job$frame), as.list(job$measures), job$paired)
    out <- list(tables = setNames(list(), character()), notes = setNames(list(), character()))
    for (name in names(stats)) {
        if (is.data.frame(stats[[name]])) {
            path <- file.path(job$folder, paste0(length(out$tables), ".csv"))
            write.csv(flat(stats[[name]]), path, row.names = TRUE)
            out$tables[[name]] <- path
        } else {
            out$notes[[name]] <- paste(stats[[name]], collapse = "\n")
        }
    }

} else {
    stop("unknown task: ", job$task)
}

write_json(out, job$result, auto_unbox = TRUE, null = "null", na = "null", digits = NA)
