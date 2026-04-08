# Set advanced parameters for RunHarmony

Set advanced parameters for RunHarmony

## Usage

``` r
harmony_options(
  alpha = 0.2,
  tau = 0,
  block.size = 0.05,
  max.iter.cluster = 4,
  epsilon.cluster = 0.001,
  epsilon.harmony = 0.01,
  batch.prop.cutoff = 1e-05
)
```

## Arguments

- alpha:

  When setting lambda = NULL and use lambda estimation mode, lambda
  would be determined by the expected number of cells assuming
  idependece between batches and clusters. i.e., lambda = alpha \*
  expected number of cells, default 0.2 and alpha should be 0 \< alpha
  \< 1

- tau:

  Protection against overclustering small datasets with large ones.
  \`tau\` is the expected number of cells per cluster.

- block.size:

  What proportion of cells to update during clustering. Between 0 to 1,
  default 0.05. Larger values may be faster but less accurate.

- max.iter.cluster:

  Maximum number of rounds to run clustering at each round of Harmony.

- epsilon.cluster:

  Convergence tolerance for clustering round of Harmony. Set to -Inf to
  never stop early.

- epsilon.harmony:

  Convergence tolerance for Harmony. Set to -Inf to never stop early.
  When \`epsilon.harmony\` is set to not NULL, then user-supplied values
  of \`early_stop\` is ignored.

- batch.prop.cutoff:

  During the integration step, if a batch has less of the specified
  proportion in a harmony cluster it will be excluded from the
  integration step. For example, batch.prop.cutoff=0.01 and a batch has
  less than 1/100 of its cells soft-assigned to a cluster this batch
  won't participating in the correction step for the particular batch.

## Value

Return a list for \`.options\` argument of \`RunHarmony\`

## Examples

``` r
## If want to set max.iter.cluster to be 100, do
if (FALSE) { # \dontrun{
RunHarmony(data_meta, meta_data, vars_use,
              .options = harmony_options(max.iter.cluster = 100))
} # }
```
