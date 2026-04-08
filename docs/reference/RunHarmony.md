# Generic function that runs the harmony algorithm on single-cell genomics cell embeddings.

RunHarmony is generic function that runs the main Harmony algorithm. If
working with single cell R objects, please refer to the documentation of
the appropriate generic API:
([`RunHarmony.Seurat()`](https://pati-ni.github.io/harmony/reference/RunHarmony.Seurat.md)
or
[`RunHarmony.SingleCellExperiment()`](https://pati-ni.github.io/harmony/reference/RunHarmony.SingleCellExperiment.md)).
If users work with other forms of cell embeddings, the can pass them
directly to harmony using
[`RunHarmony.default()`](https://pati-ni.github.io/harmony/reference/RunHarmony.default.md)
API. All the function arguments listed here are common in all RunHarmony
interfaces.

## Usage

``` r
RunHarmony(...)
```

## Arguments

- ...:

  Arguments passed on to
  [`RunHarmony.default`](https://pati-ni.github.io/harmony/reference/RunHarmony.default.md)

  `theta`

  :   Diversity clustering penalty parameter. Specify for each variable
      in vars_use Default theta=2. theta=0 does not encourage any
      diversity. Larger values of theta result in more diverse clusters.

  `sigma`

  :   Width of soft kmeans clusters. Default sigma=0.1. Sigma scales the
      distance from a cell to cluster centroids. Larger values of sigma
      result in cells assigned to more clusters. Smaller values of sigma
      make soft kmeans cluster approach hard clustering.

  `lambda`

  :   Ridge regression penalty. Default lambda=1. Bigger values protect
      against over correction. If several covariates are specified, then
      lambda can also be a vector which needs to be equal length with
      the number of variables to be corrected. In this scenario, each
      covariate level group will be assigned the scalars specified by
      the user. If set to NULL, harmony will start lambda estimation
      mode to determine lambdas automatically and try to minimize
      overcorrection (Use with caution still in beta testing).

  `nclust`

  :   Number of clusters in model. nclust=1 equivalent to simple linear
      regression.

  `max_iter`

  :   Maximum number of rounds to run Harmony. One round of Harmony
      involves one clustering and one correction step.

  `early_stop`

  :   Enable early stopping for harmony. The harmonization process will
      stop when the change of objective function between corrections
      drops below 1e-4

  `ncores`

  :   Number of processors to be used for math operations when optimized
      BLAS is available. If BLAS is not supporting multithreaded then
      this option has no effect. By default, ncore=1 which runs as a
      single-threaded process. Although Harmony supports multiple cores,
      it is not optimized for multithreading. Increase this number for
      large datasets iff single-core performance is not adequate.

  `plot_convergence`

  :   Whether to print the convergence plot of the clustering objective
      function. TRUE to plot, FALSE to suppress. This can be useful for
      debugging.

  `verbose`

  :   Whether to print progress messages. TRUE to print, FALSE to
      suppress.

  `.options`

  :   Setting advanced parameters of RunHarmony. This must be the result
      from a call to \`harmony_options\`. See ?\`harmony_options\` for
      parameters not listed above and more details.

## Value

If used with single-cell objects, it will return the updated single-sell
object. For standalone operation, it returns the corrected cell
embeddings or the R6 harmony object (see
[`RunHarmony.default()`](https://pati-ni.github.io/harmony/reference/RunHarmony.default.md)).

## See also

Other RunHarmony:
[`RunHarmony.Seurat()`](https://pati-ni.github.io/harmony/reference/RunHarmony.Seurat.md),
[`RunHarmony.SingleCellExperiment()`](https://pati-ni.github.io/harmony/reference/RunHarmony.SingleCellExperiment.md),
[`RunHarmony.default()`](https://pati-ni.github.io/harmony/reference/RunHarmony.default.md)
