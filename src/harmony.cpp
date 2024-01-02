#include <algorithm>
#include <chrono>

#include "harmony.h"
#include "types.h"
#include "utils.h"




harmony::harmony() :
    window_size(3),
    ran_setup(false),
    ran_init(false),
    lambda_estimation(false),
    verbose(false)
    
{}



void harmony::setup(const RMAT& __Z, const RSPMAT& __Phi,
                    const RVEC __sigma, const RVEC __theta, const RVEC __lambda, const float __alpha, const int __max_iter_kmeans,
                    const float __epsilon_kmeans, const float __epsilon_harmony,
                    const int __K, const float __block_size,
                    const std::vector<int>& __B_vec, const bool __verbose) {
    
  // Algorithm constants
  N = __Z.n_cols;
  B = __Phi.n_rows;
  d = __Z.n_rows;
  
  Z_orig = conv_to<MATTYPE>::from(__Z);
  Z_corr = arma::normalise(Z_orig, 2, 0);

  
  Phi = conv_to<SPMAT>::from(__Phi);
  Phi_t = Phi.t();
  
  // Create index
  std::vector<unsigned>counters;
  VECTYPE sizes(sum(Phi, 1));
  // std::cout << sizes << std::endl;
  for (unsigned i = 0; i < sizes.n_elem; i++) {
    arma::uvec a(int(sizes(i)));
    index.push_back(a);
    counters.push_back(0);
  }

  SPMAT::const_iterator it =     Phi.begin();
  SPMAT::const_iterator it_end = Phi.end();
  for(; it != it_end; ++it)
  {
    unsigned int row_idx = it.row();
    unsigned int col_idx = it.col();
    index[row_idx](counters[row_idx]++) = col_idx;
  }

  Pr_b = sum(Phi, 1) / N;

  
  epsilon_kmeans = __epsilon_kmeans;
  epsilon_harmony = __epsilon_harmony;

  // Hyperparameters
  K = __K;
  if (__lambda(0) == -1) {
    lambda_estimation = true;
  } else {
    lambda = conv_to<VECTYPE>::from(__lambda);
  }
  B_vec = __B_vec;
  sigma = conv_to<VECTYPE>::from(__sigma);

  if(__Z.n_cols < 6) {
    std::string error_message = "Refusing to run with less than 6 cells";
    Rcpp::stop(error_message);
  } else if (__Z.n_cols < 40) {
    Rcpp::warning("Too few cells. Setting block_size to 0.2");
    block_size = 0.2;
  } else {
    block_size = __block_size;
  } 
  theta = conv_to<VECTYPE>::from(__theta);
  max_iter_kmeans = __max_iter_kmeans;

  verbose = __verbose;
  
  allocate_buffers();
  ran_setup = true;

  alpha = __alpha;
  
  
}


void harmony::allocate_buffers() {
  
  dist_mat = zeros<MATTYPE>(K, N);
  O = E = zeros<MATTYPE>(K, B);
  
  // Hack: create matrix of ones by creating zeros and then add one!
  SPMAT intcpt = zeros<SPMAT>(1, N);
  intcpt = intcpt+1;
  
  Phi_moe = join_cols(intcpt, Phi);
  Phi_moe_t = Phi_moe.t();


  W = zeros<MATTYPE>(B + 1, d);
}


void harmony::init_cluster_cpp() {
  
  Y = kmeans_centers(Z_corr, K, verbose);

  // Cosine normalization of data centrods
  Y = arma::normalise(Y, 2, 0);

  // (2) ASSIGN CLUSTER PROBABILITIES
  // using a nice property of cosine distance,
  // compute squared distance directly with cross product
  dist_mat = 2 * (1 - Y.t() * Z_corr);
  
  R = -dist_mat;
  R.each_col() /= sigma;
  R = exp(R);
  R.each_row() /= sum(R, 0);
  
  // (3) BATCH DIVERSITY STATISTICS
  E = sum(R, 1) * Pr_b.t();
  O = R * Phi_t;
  
  compute_objective();
  objective_harmony.push_back(objective_kmeans.back());

  ran_init = true;
}

void harmony::compute_objective() {
  const float norm_const = 2000/((float)N);
  float kmeans_error = as_scalar(accu(R % dist_mat));  
  float _entropy = as_scalar(accu(safe_entropy(R).each_col() % sigma)); // NEW: vector sigma
  float _cross_entropy = as_scalar(
      accu((R.each_col() % sigma) % ((arma::repmat(theta.t(), K, 1) % log((O + E) / E)) * Phi)));

  // Push back the data
  objective_kmeans.push_back((kmeans_error + _entropy + _cross_entropy) * norm_const);
  objective_kmeans_dist.push_back(kmeans_error * norm_const);
  objective_kmeans_entropy.push_back(_entropy * norm_const);
  objective_kmeans_cross.push_back(_cross_entropy * norm_const);
}


bool harmony::check_convergence(int type) {
  float obj_new, obj_old;
  switch (type) {
    case 0: 
      // Clustering 
      // compute new window mean
      obj_old = 0;
      obj_new = 0;
      for (unsigned i = 0; i < window_size; i++) {
        obj_old += objective_kmeans[objective_kmeans.size() - 2 - i];
        obj_new += objective_kmeans[objective_kmeans.size() - 1 - i];
      }
      if ((obj_old - obj_new) / abs(obj_old) < epsilon_kmeans) {
        return(true); 
      } else {
        return(false);
      }
    case 1:
      // Harmony
      obj_old = objective_harmony[objective_harmony.size() - 2];
      obj_new = objective_harmony[objective_harmony.size() - 1];
      if ((obj_old - obj_new) / abs(obj_old) < epsilon_harmony) {
        return(true);              
      } else {
        return(false);              
      }
  }
  
  // gives warning if we don't give default return value
  return(true);
}


int harmony::cluster_cpp() {
  int err_status = 0;
  Progress p(max_iter_kmeans, verbose);
  unsigned iter;


  Z_corr = arma::normalise(Z_corr, 2, 0);
  // Z_corr has changed
  // R has assumed to not change
  // so update Y to match new integrated data  
  for (iter = 0; iter < max_iter_kmeans; iter++) {
      
      p.increment();
      if (Progress::check_abort())
	  return(-1);
    
      // STEP 1: Update Y (cluster centroids)
      Y = arma::normalise(Z_corr * R.t(), 2, 0);

      dist_mat = 2 * (1 - Y.t() * Z_corr); // Y was changed
              
      // STEP 3: Update R    
      err_status = update_R();
      if (err_status != 0) {
	  // Rcout << "Compute R failed. Exiting from clustering." << endl;
	  return err_status;
      }
    
      // STEP 4: Check for convergence
      compute_objective();
    
      if (iter > window_size) {
	  bool convergence_status = check_convergence(0);
	  if (convergence_status) {
	      iter++;
	      break;
	  }
      }
  }
  
  kmeans_rounds.push_back(iter);
  objective_harmony.push_back(objective_kmeans.back());
  return 0;
}






int harmony::update_R() {

  // Generate the 0,N-1 indices
  uvec indices = linspace<uvec>(0, N - 1, N);
  update_order = shuffle(indices);
  
  // Inverse index
  uvec reverse_index(N, arma::fill::zeros);
  reverse_index.rows(update_order) = indices;    

  // GENERAL CASE: online updates, in blocks of size (N * block_size)
  unsigned n_blocks = (int)(my_ceil(1.0 / block_size));
  unsigned cells_per_block = unsigned(N * block_size);
  
  // Reference matrices to avoid allocating memory
  MATTYPE& R_randomized = R;
  R_randomized = R_randomized.cols(update_order);
  
  MATTYPE& dist_mat_randomized = dist_mat;   
  dist_mat_randomized = dist_mat_randomized.cols(update_order);
  
  SPMAT Phi_randomized(Phi.cols(update_order));
  SPMAT Phi_t_randomized(Phi_randomized.t());

  for (unsigned i = 0; i < n_blocks; i++) {
    unsigned idx_min = i*cells_per_block;
    unsigned idx_max = ((i+1) * cells_per_block) - 1; // - 1 because of submat
    if (i == n_blocks-1) {
      // we are in the last block, so include everything. Up to 19
      // extra cells.
      idx_max = N - 1;
    }

    auto Rcells = R_randomized.submat(0, idx_min, R_randomized.n_rows - 1, idx_max);
    auto Phicells = Phi_randomized.submat(0, idx_min, Phi_randomized.n_rows - 1, idx_max);
    auto Phi_tcells = Phi_t_randomized.submat(idx_min, 0, idx_max, Phi_t_randomized.n_cols - 1);
    auto dist_matcells = dist_mat_randomized.submat(0, idx_min, dist_mat_randomized.n_rows - 1, idx_max);

    // Step 1: remove cells
    E -= sum(Rcells, 1) * Pr_b.t();
    O -= Rcells * Phi_tcells;

    // Step 2: recompute R for removed cells
    Rcells = -dist_matcells;
    Rcells.each_col() /= sigma; // NEW: vector sigma
    Rcells = exp(Rcells);
    Rcells = arma::normalise(Rcells, 1, 0);
    Rcells = Rcells % (harmony_pow(E/(O + E), theta) * Phicells);
    Rcells = arma::normalise(Rcells, 1, 0); // L1 norm columns


    // Step 3: put cells back 
    E += sum(Rcells, 1) * Pr_b.t();
    O += Rcells * Phi_tcells;
  }
  
  // Unshuffle R (this updates also the class objects since this is a
  // reference to these class attributes)
  R_randomized = R_randomized.cols(reverse_index);
  dist_mat = dist_mat.cols(reverse_index);  
  
  return 0;
}


void harmony::moe_correct_ridge_cpp() {
  
  SPMAT _Rk(N, N);
  SPMAT lambda_mat(B + 1, B + 1);

  if(!lambda_estimation) {
    // Set lambda if we have to
    lambda_mat.diag() = lambda;
  }
  Z_corr = Z_orig;
  Progress p(K, verbose);
  for (unsigned k = 0; k < K; k++) {
    p.increment();
    if (Progress::check_abort())
      return;
    if (lambda_estimation) {
      lambda_mat.diag() = find_lambda_cpp(alpha, E.row(k).t());
    }
    _Rk.diag() = R.row(k);
    SPMAT Phi_Rk = Phi_moe * _Rk;
    
    MATTYPE inv_cov(arma::inv(MATTYPE(Phi_Rk * Phi_moe_t + lambda_mat)));

    // Calculate R-scaled PCs once
    MATTYPE Z_tmp = Z_orig.each_row() % R.row(k);
    
    // Generate the betas contribution of the intercept using the data
    // This erases whatever was written before in W
    W = inv_cov.unsafe_col(0) * sum(Z_tmp, 1).t();

    // Calculate betas by calculating each batch contribution
    for(unsigned b=0; b < B; b++) {
      // inv_conv is B+1xB+1 whereas index is B long
      W += inv_cov.unsafe_col(b+1) * sum(Z_tmp.cols(index[b]), 1).t();
    }
    
    W.row(0).zeros(); // do not remove the intercept
    Z_corr -= W.t() * Phi_Rk;
  }

}

// CUBETYPE harmony::moe_ridge_get_betas_cpp() {
//   CUBETYPE W_cube(B+1, d, K); // rows, cols, slices

//   SPMAT _Rk(N, N);
//   SPMAT lambda_mat(B + 1, B + 1);

//   if (!lambda_estimation) {
//     // Set lambda if we have to
//     lambda_mat.diag() = lambda;
//   }

//   for (unsigned k = 0; k < K; k++) {
//       _Rk.diag() = R.row(k);
//       if (lambda_estimation){
//         lambda_mat.diag() = find_lambda_cpp(alpha, E.row(k).t());
//       }
//       SPMAT Phi_Rk = Phi_moe * _Rk;
//       W_cube.slice(k) = arma::inv(MATTYPE(Phi_Rk * Phi_moe_t + lambda_mat)) * Phi_Rk * Z_orig.t();
//   }

//   return W_cube;
// }

RCPP_MODULE(harmony_module) {
  class_<harmony>("harmony")
      .constructor()
      .field("Z_corr", &harmony::Z_corr)
      .field("Z_orig", &harmony::Z_orig)
      // .field("Phi", &harmony::Phi)
      // .field("Phi_moe", &harmony::Phi_moe)
      .field("N", &harmony::N)
      .field("B", &harmony::B)
      .field("K", &harmony::K)
      .field("d", &harmony::d)
      .field("O", &harmony::O)
      .field("E", &harmony::E)
      .field("Y", &harmony::Y)
      .field("Pr_b", &harmony::Pr_b)
      .field("W", &harmony::W)
      .field("R", &harmony::R)
      .field("theta", &harmony::theta)
      .field("sigma", &harmony::sigma)
      .field("lambda", &harmony::lambda)
      .field("kmeans_rounds", &harmony::kmeans_rounds)
      .field("objective_kmeans", &harmony::objective_kmeans)
      .field("objective_kmeans_dist", &harmony::objective_kmeans_dist)
      .field("objective_kmeans_entropy", &harmony::objective_kmeans_entropy)
      .field("objective_kmeans_cross", &harmony::objective_kmeans_cross)    
      .field("objective_harmony", &harmony::objective_harmony)
      .field("max_iter_kmeans", &harmony::max_iter_kmeans)
      .method("check_convergence", &harmony::check_convergence)
      .method("setup", &harmony::setup)
      .method("compute_objective", &harmony::compute_objective)
      .method("init_cluster_cpp", &harmony::init_cluster_cpp)
      .method("cluster_cpp", &harmony::cluster_cpp)	  
      .method("moe_correct_ridge_cpp", &harmony::moe_correct_ridge_cpp)
      // .method("moe_ridge_get_betas_cpp", &harmony::moe_ridge_get_betas_cpp)
      .field("B_vec", &harmony::B_vec)
      .field("alpha", &harmony::alpha)
      ;
}
