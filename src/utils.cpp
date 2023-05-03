#include "utils.h"
#include "types.h"

//[[Rcpp::export]]
arma::mat kmeans_centers(const arma::mat& X, const int K){
  
  // Environment 
  Rcpp::Environment stats_env("package:stats");
  // Cast function as callable from C++
  Rcpp::Function kmeans = stats_env["kmeans"];
  // Call the function and receive its list output
  Rcpp::List res = kmeans(Rcpp::_["x"] = X.t(),
                          Rcpp::_["centers"] = K,
                          Rcpp::_["iter.max"] = 25,
                          Rcpp::_["nstart"] = 10
                          );
  return res["centers"];
}


MATTYPE safe_entropy(const MATTYPE& X) {
  MATTYPE A = X % log(X);
  A.elem(find_nonfinite(A)).zeros();
  return(A);
}

// Overload pow to work on a MATTYPErix and vector
MATTYPE harmony_pow(MATTYPE A, const VECTYPE& T) {

  for (unsigned c = 0; c < A.n_cols; c++) {
    A.unsafe_col(c) = pow(A.unsafe_col(c), as_scalar(T.row(c)));
  }
  return(A);
}

VECTYPE calculate_norm(const MATTYPE& M){
  VECTYPE x(M.n_cols);
  for(unsigned i = 0; i < M.n_cols; i++){
    x(i) = norm(M.col(i));
  }
  return x;
}


//https://stackoverflow.com/questions/8377412/ceil-function-how-can-we-implement-it-ourselves
int my_ceil(float num) {
    int inum = (int)num;
    if (num == (float)inum) {
        return inum;
    }
    return inum + 1;
}


arma::mat estimate_residuals(const arma::mat& O, const arma::mat& E){
    int B = O.n_cols, K = O.n_rows;
    // Intercept E, one-hot k, one-hot b
    const int dof=2;
    // Subtract from the design matrix the degrees of freedom of the parameters!
    arma::mat design = arma::zeros(K*B, 2 + B + K - dof);
    
    // Assign one to intercepts
    design.col(0) = arma::ones(design.n_rows);
    // Assign E to the first
    design.col(1) = arma::vectorise(E);
    
    auto O_flat = arma::vectorise(O);
    
    // Get estimates
    for(int b = 0; b < B; b++) {
	for(int k = 0; k < K; k++) {
	    int index = (b * K) + k;
	    if(b!=0){
		design(index, 2 + b - 1) = 1;
	    }
	    if(k !=0){
		design(index, 2 + B - 1 + k - 1) = 1;		
	    }	    
	}
    }
    
    // Perform linear regression
    arma::mat spectra = arma::inv(design.t() * design);
    arma::mat betas = spectra * design.t() * O_flat;
    arma::mat O_est = design * betas;
    
    // Return estimations
    return arma::reshape(O_est, K, B);
}
