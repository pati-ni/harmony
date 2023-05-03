#pragma once
#include "types.h"
#include <RcppArmadillo.h>

arma::mat kmeans_centers(const arma::mat& X, const int K);

MATTYPE safe_entropy(const MATTYPE& X);

MATTYPE harmony_pow(MATTYPE A, const VECTYPE& T);

VECTYPE calculate_norm(const MATTYPE& M);


int my_ceil(float num);


arma::mat estimate_residuals(const arma::mat& O, const arma::mat& E);
