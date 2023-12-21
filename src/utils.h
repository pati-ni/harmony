#pragma once
#include "types.h"
#include <RcppArmadillo.h>

MATTYPE kmeans_centers(const MATTYPE& X, const int K);

MATTYPE safe_entropy(const MATTYPE& X);

MATTYPE harmony_pow(MATTYPE A, const VECTYPE& T);

VECTYPE calculate_norm(const MATTYPE& M);


int my_ceil(float num);


VECTYPE find_lambda_cpp(const float alpha, const VECTYPE& cluster_E);
