# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Harmony is an R package for fast, sensitive integration of single-cell genomics data. It corrects batch effects in PCA embeddings using a mixture-of-experts ridge regression with soft k-means clustering.

## Build and development commands

All commands are run from within R (or via `Rscript`):

```r
# Install dependencies and build the package (including C++ compilation)
devtools::install_local(".", force=TRUE, upgrade=FALSE)

# Or rebuild just the C++ shared library
devtools::load_all(".")

# Run all tests
devtools::test()

# Run a single test file
testthat::test_file("tests/testthat/test_integration.R")

# Generate documentation from roxygen2 comments
devtools::document()

# Full R CMD CHECK
devtools::check()
```

From the shell, you can also run:
```sh
R CMD INSTALL .
R CMD check .
```

## Architecture

### Hybrid R/C++ design

The package has two layers:

1. **C++ core** (`src/`): A `harmony` C++ class implementing the algorithm, compiled as a shared library via Rcpp. The class is exposed to R through an Rcpp module (`RCPP_MODULE(harmony_module)` in `src/harmony.cpp`). R code instantiates it with `new(harmony)` and calls methods directly.

2. **R layer** (`R/`): S3 dispatch, input validation, data preparation, and integration with Seurat/SingleCellExperiment objects. All public API lives here.

### Execution flow

`RunHarmony()` (S3 generic in `R/RunHarmony.R`) dispatches to:
- `RunHarmony.default()` in `R/ui.R` — main implementation
- `RunHarmony.Seurat()` / `RunHarmony.SingleCellExperiment()` in `R/RunHarmony.R` — extract embeddings and delegate to `default`

Inside `RunHarmony.default()`:
1. Validates inputs, builds a sparse design matrix `phi` encoding batch membership
2. Constructs the C++ `harmony` object and calls `harmonyObj$setup(...)`, then `harmonyObj$init_cluster_cpp()`
3. Calls `harmonize()` (in `R/utils.R`), which loops over `cluster_cpp()` → `moe_correct_ridge_cpp()` → `check_convergence()` until convergence

The C++ `harmony` class (`src/harmony.h`, `src/harmony.cpp`) owns all numeric state:
- `Z_orig` / `Z_corr`: original and batch-corrected PCA embeddings (d × N)
- `R`: soft cluster assignments (K × N)
- `Y`: cluster centroids (d × K)
- `O`, `E`: observed/expected batch-cluster co-occurrence matrices (K × B)
- `Phi` / `Phi_moe`: sparse batch-membership matrices

### Type conventions

All C++ math uses **single-precision float** (`SCALAR = float`, defined in `src/types.h`). The `RMAT`/`RSPMAT`/`RVEC` types are double-precision Armadillo types used only at the R↔C++ boundary for conversion.

### Advanced options

`harmony_options()` (in `R/harmony_option.R`) controls internal algorithm parameters (`block.size`, `max.iter.cluster`, convergence tolerances, etc.) and is passed via the `.options` argument to `RunHarmony`. Legacy top-level arguments (e.g., `block.size`, `tau`) were deprecated; warn users to use `.options` instead.

### OpenMP / multithreading

OpenMP support is **disabled by default** — the relevant line in `src/Makevars` is commented out:
```
# PKG_CXXFLAGS = $(SHLIB_OPENMP_CXXFLAGS)
```
See `PERFORMANCE.md` for instructions on enabling it. Thread count for BLAS operations is controlled at runtime via the `ncores` parameter (uses `RhpcBLASctl`).
