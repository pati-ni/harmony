# This is the primary harmony interface.

Use this generic with a cell embeddings matrix, a metadata table and a
categorical covariate to run the Harmony algorithm directly on cell
embedding matrix.

## Usage

``` r
# Default S3 method
RunHarmony(
  data_mat,
  meta_data,
  vars_use,
  theta = NULL,
  sigma = 0.1,
  lambda = NULL,
  nclust = NULL,
  max_iter = 10,
  early_stop = TRUE,
  ncores = 1,
  plot_convergence = FALSE,
  return_object = FALSE,
  verbose = TRUE,
  .options = harmony_options(),
  ...
)
```

## Arguments

- data_mat:

  Matrix of cell embeddings. Cells can be rows or columns and will be
  inferred by the rows of meta_data.

- meta_data:

  Either (1) Dataframe with variables to integrate or (2) vector with
  labels.

- vars_use:

  If meta_data is dataframe, this defined which variable(s) to remove
  (character vector).

- theta:

  Diversity clustering penalty parameter. Specify for each variable in
  vars_use Default theta=2. theta=0 does not encourage any diversity.
  Larger values of theta result in more diverse clusters.

- sigma:

  Width of soft kmeans clusters. Default sigma=0.1. Sigma scales the
  distance from a cell to cluster centroids. Larger values of sigma
  result in cells assigned to more clusters. Smaller values of sigma
  make soft kmeans cluster approach hard clustering.

- lambda:

  Ridge regression penalty. Default lambda=1. Bigger values protect
  against over correction. If several covariates are specified, then
  lambda can also be a vector which needs to be equal length with the
  number of variables to be corrected. In this scenario, each covariate
  level group will be assigned the scalars specified by the user. If set
  to NULL, harmony will start lambda estimation mode to determine
  lambdas automatically and try to minimize overcorrection (Use with
  caution still in beta testing).

- nclust:

  Number of clusters in model. nclust=1 equivalent to simple linear
  regression.

- max_iter:

  Maximum number of rounds to run Harmony. One round of Harmony involves
  one clustering and one correction step.

- early_stop:

  Enable early stopping for harmony. The harmonization process will stop
  when the change of objective function between corrections drops below
  1e-4

- ncores:

  Number of processors to be used for math operations when optimized
  BLAS is available. If BLAS is not supporting multithreaded then this
  option has no effect. By default, ncore=1 which runs as a
  single-threaded process. Although Harmony supports multiple cores, it
  is not optimized for multithreading. Increase this number for large
  datasets iff single-core performance is not adequate.

- plot_convergence:

  Whether to print the convergence plot of the clustering objective
  function. TRUE to plot, FALSE to suppress. This can be useful for
  debugging.

- return_object:

  (Advanced Usage) Whether to return the Harmony object or only the
  corrected PCA embeddings.

- verbose:

  Whether to print progress messages. TRUE to print, FALSE to suppress.

- .options:

  Setting advanced parameters of RunHarmony. This must be the result
  from a call to \`harmony_options\`. See ?\`harmony_options\` for
  parameters not listed above and more details.

- ...:

  other parameters that are not part of the API

## Value

By default, matrix with corrected PCA embeddings. If return_object is
TRUE, returns the full Harmony object (R6 reference class type).

## See also

Other RunHarmony:
[`RunHarmony()`](https://pati-ni.github.io/harmony/reference/RunHarmony.md),
[`RunHarmony.Seurat()`](https://pati-ni.github.io/harmony/reference/RunHarmony.Seurat.md),
[`RunHarmony.SingleCellExperiment()`](https://pati-ni.github.io/harmony/reference/RunHarmony.SingleCellExperiment.md)

## Examples

``` r


## By default, Harmony inputs a cell embedding matrix
if (FALSE) { # \dontrun{
harmony_embeddings <- RunHarmony(cell_embeddings, meta_data, 'dataset')
} # }

## If PCA is the input, the PCs need to be scaled
data(cell_lines_small)
pca_matrix <- cell_lines_small$scaled_pcs
meta_data <- cell_lines_small$meta_data
harmony_embeddings <- RunHarmony(pca_matrix, meta_data, 'dataset')
#> Transposing data matrix
#> Using automatic lambda estimation
#> Thetas: 2
#> Initializing state using k-means centroids initialization
#> Initializing centroids
#> Harmony 1/10
#> Harmony converged after 1 iterations

## Output is a matrix of corrected PC embeddings
dim(harmony_embeddings)
#> [1] 300  20
harmony_embeddings[seq_len(5), seq_len(5)]
#>                X1            X2            X3            X4            X5
#> [1,]  0.007770998 -0.0044285185 -2.658067e-03 -0.0079813693 -0.0001508195
#> [2,] -0.011452863  0.0007276849 -2.526561e-04 -0.0015449575  0.0034087112
#> [3,]  0.009720036 -0.0083047664 -9.797361e-05 -0.0042609675 -0.0007274408
#> [4,]  0.009570926 -0.0030485140  7.086939e-04 -0.0075727096 -0.0012533891
#> [5,] -0.009786356  0.0028862588 -3.220626e-03 -0.0002428817  0.0008977586

## Finally, we can return an object with all the underlying data structures
harmony_object <- RunHarmony(pca_matrix, meta_data, 'dataset', return_object=TRUE)
#> Transposing data matrix
#> Using automatic lambda estimation
#> Thetas: 2
#> Initializing state using k-means centroids initialization
#> Initializing centroids
#> Harmony 1/10
#> Harmony converged after 1 iterations
dim(harmony_object$Y) ## cluster centroids
#> [1] 20 10
dim(harmony_object$R) ## soft cluster assignment
#> [1]  10 300
dim(harmony_object$getZcorr()) ## corrected PCA embeddings
#> [1]  20 300
head(harmony_object$O) ## batch by cluster co-occurence matrix
#>             [,1]         [,2]         [,3]
#> [1,] 20.79351997 1.927618e+00 6.266189e-06
#> [2,] 15.79600239 3.747352e-08 2.994562e+01
#> [3,]  1.08318412 1.681248e+01 2.187659e-07
#> [4,] 10.19679642 7.671215e-07 1.904781e+01
#> [5,]  0.08097884 2.317920e+01 1.588407e-07
#> [6,] 21.26292419 4.868720e-01 1.358413e-06
```
