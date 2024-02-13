#include "types.h"
#include <Rcpp.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <progress.hpp>

using namespace Rcpp;
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppProgress)]]

using namespace std;

class harmony;
RCPP_EXPOSED_CLASS(harmony)
  
  
#include "harmony_types.h"
  
class harmony { 
public:
  
  harmony();
  
  void setup(const RMAT& __Z, const RSPMAT& __Phi,
	     const RVEC __sigma, const RVEC __theta,
	     const RVEC __lambda, const float __alpha, const int __max_iter_kmeans,
	     const float __epsilon_kmeans, const float __epsilon_harmony,
	     const int __K, const float __block_size, 
	     const vector<int>& __B_vec, const bool __verbose);
  
  /* METHODS */
  void moe_correct_ridge_cpp();
  // CUBETYPE moe_ridge_get_betas_cpp();
  int cluster_cpp();

  void init_cluster_cpp();
  void allocate_buffers();
  void compute_objective(); 
  int update_R();
  bool check_convergence(int type);
  void setY(const MATTYPE& Z);

  /* FIELDS */
  MATTYPE R, Z_orig, Z_corr, Y;
  SPMAT Phi, Phi_moe, Phi_moe_t, Phi_t, Rk;
  VECTYPE Pr_b, theta, N_b, sigma, lambda;

  // auxilary data structures
  vector<float> objective_kmeans, objective_kmeans_dist, objective_kmeans_entropy, objective_kmeans_cross, objective_harmony;
  vector<int> kmeans_rounds, B_vec; // OLD: Kb
  std::vector<arma::uvec>index;
  
  float block_size, epsilon_kmeans, epsilon_harmony, alpha;
  unsigned int N, K, B, d, max_iter_kmeans, window_size;

  // buffers
  MATTYPE W, _scale_dist, dist_mat, O, E, dir_prior; // N_k, N_kb, N_b, numerator, denominator, C;
  uvec update_order, cells_update;
  

  // flags
  bool ran_setup, ran_init, lambda_estimation,  verbose; // do_merge_R;
  
};

  
