#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <cstring>

#include "harmony.h"
#include "types.h"
#include "utils.h"

#include "timer.h"

void print_timers() {  
  std::cerr << "Print timers" << std::endl;
  for(auto const& t : timers) {
    std::cerr << t.first << " I got " << t.second.iter << " iterations " << t.second.elapsed/1000 << " seconds" << std::endl;
  }
}


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
                    const std::vector<int>& __B_vec, float __batch_proportion_cutoff, const bool __verbose) {
    
  // Algorithm constants
  N = __Z.n_cols;
  B = __Phi.n_rows;
  d = __Z.n_rows;

  
  
  if (false) {
    // TODO(Nikos) Sort according to one covariate to optimize
    // correction steps using submat operations later.
    Timer t(timers["sorting_elems"]);        
    orig_index.reserve(N);
    RSPMAT::const_iterator it =     __Phi.begin();
    RSPMAT::const_iterator it_end = __Phi.end();
    batch_indptr = arma::zeros<uvec>(B+1);
    for (unsigned i=0; it != it_end; ++it, i++)
    {
      unsigned int row_idx = it.row(); // Batch
      unsigned int col_idx = it.col(); // Cell
      batch_indptr(row_idx+1) += 1; // i+1, indptr
      orig_index.push_back({row_idx, col_idx});
      
      if (col_idx != i) {
	Rcpp::stop("Misalignment during sorting");
      }
    }
    // Update batch_indptr
    for (unsigned i=1; i < B+1;++i) {
      batch_indptr(i) += batch_indptr(i-1);
    }
    
    std::sort(orig_index.begin(), orig_index.end(), CellEntryCompare);
    new_index = arma::uvec(orig_index.size());
    unsigned i = 0;
    for (const auto& s : orig_index) {
      new_index(i++) = s.cell_number;
    }
    
    uvec indices = linspace<uvec>(0, N - 1, N);
    // Inverse index
    original_index = arma::uvec(size(indices));
    original_index.rows(new_index) = indices;
    Phi_t = SPMAT(indices,
		  batch_indptr,
		  VECTYPE(indices.n_rows, arma::fill::ones),
		  N,
		  B);
    Z_orig = conv_to<MATTYPE>::from(__Z.cols(new_index));
  } else{
    original_index = linspace<uvec>(0, N - 1, N);
    Z_orig = conv_to<MATTYPE>::from(__Z);
  }     
  
  
  Z_corr = arma::normalise(Z_orig, 2, 0);
  
  Phi = conv_to<SPMAT>::from(__Phi);
  Phi_t = Phi.t();
  // Phi = Phi_t.t();
  
  
  // Create index
  std::vector<unsigned>counters;
  
  batch_sizes = conv_to<arma::uvec>::from(VECTYPE(sum(Phi, 1)));
  for (unsigned i = 0; i < batch_sizes.n_elem; i++) {
    arma::uvec a(batch_sizes(i));
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

  
  
  // Covariate bounds
  if (B_vec.size() > 1) {
    covariate_bounds.resize(B_vec.size() - 1 );
    std::partial_sum(B_vec.begin(), B_vec.end(), covariate_bounds.begin(), std::plus<unsigned>());
  } else {
    covariate_bounds.push_back(B_vec.front());
  }
  
  
  theta = conv_to<VECTYPE>::from(__theta);
  max_iter_kmeans = __max_iter_kmeans;

  verbose = __verbose;
  
  allocate_buffers();
  ran_setup = true;

  alpha = __alpha;
  batch_proportion_cutoff = __batch_proportion_cutoff;
  
  
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
  float kmeans_error = as_scalar(my_accu(R % dist_mat));  
  float _entropy = as_scalar(my_accu(safe_entropy(R).each_col() % sigma)); // NEW: vector sigma
  float _cross_entropy = as_scalar(
      my_accu((R.each_col() % sigma) % ((arma::repmat(theta.t(), K, 1) % log((O + E) / E)) * Phi)));

  // Push back the data
  objective_kmeans.push_back((kmeans_error + _entropy + _cross_entropy) * norm_const);
  objective_kmeans_dist.push_back(kmeans_error * norm_const);
  objective_kmeans_entropy.push_back(_entropy * norm_const);
  objective_kmeans_cross.push_back(_cross_entropy * norm_const);
  
  std::cout << "Objective: \n\t Harmony:\t" << objective_kmeans.back() << std::endl;
  std::cout << "\t kmeans:\t" << objective_kmeans_dist.back() << std::endl;
  std::cout << "\t soft-k:\t" << objective_kmeans_entropy.back() << std::endl;
  std::cout << "\t divers:\t" << objective_kmeans_cross.back() << std::endl;

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
      if (abs(obj_old - obj_new) / abs(obj_old) < epsilon_kmeans) {
        return(true); 
      } else {
        return(false);
      }
    case 1:
      // Harmony
      obj_old = objective_harmony[objective_harmony.size() - 2];
      obj_new = objective_harmony[objective_harmony.size() - 1];
      if ((obj_old - obj_new) / abs(obj_old) < epsilon_harmony) {
	// Unshuffle Z_corr
	
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




  if (objective_harmony.size() != 1) {
    // We are coming after a correction step and this is a cold start
    // of clustering Estimation Step. Rs are estimated from last
    // iteration's estimation and do not reflect the current Z_corr
    // embeddings. Re-estimate Rs from the new corrected parameters as
    // we did in init_cluster_cpp
    
    Z_corr = arma::normalise(Z_corr, 2, 0);
    dist_mat = 2 * (1 - Y.t() * Z_corr);  
    R = -dist_mat;
    R.each_col() /= sigma;
    R = exp(R);
    R.each_row() /= sum(R, 0);
    E = sum(R, 1) * Pr_b.t();
    O = R * Phi_t;
  }
  
  for (iter = 0; iter < max_iter_kmeans; iter++) {
      
      p.increment();
      if (Progress::check_abort())
	  return(-1);
    
      // STEP 1: Update Y (cluster centroids)
      // Y = arma::normalise(Z_corr * R.t(), 2, 0);

      // dist_mat = 2 * (1 - Y.t() * Z_corr); // Y was changed
              
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
    
    Timer *t_r = new Timer(timers["random_subset"]);
    auto Rcells = R_randomized.submat(0, idx_min, R_randomized.n_rows - 1, idx_max);
    auto Phicells = Phi_randomized.submat(0, idx_min, Phi_randomized.n_rows - 1, idx_max);
    auto Phi_tcells = Phi_t_randomized.submat(idx_min, 0, idx_max, Phi_t_randomized.n_cols - 1);
    auto dist_matcells = dist_mat_randomized.submat(0, idx_min, dist_mat_randomized.n_rows - 1, idx_max);
    delete t_r;
    
    {
      Timer t(timers["EO_update"]);
      // Step 1: remove cells
      E -= sum(Rcells, 1) * Pr_b.t();
      O -= Rcells * Phi_tcells;
    }
    // Step 2: recompute R for removed cells
    {
      Timer t(timers["Rcells_update"]);
      Rcells = -dist_matcells;
      Rcells.each_col() /= sigma; // NEW: vector sigma
      Rcells = exp(Rcells);
      Rcells = arma::normalise(Rcells, 1, 0);
      Rcells = Rcells % (harmony_pow(E / (O + E), theta) * Phicells);
      Rcells = arma::normalise(Rcells, 1, 0); // L1 norm columns
    }

    {
      Timer t(timers["EO_update"]);
      // Step 3: put cells back 
      E += sum(Rcells, 1) * Pr_b.t();
      O += Rcells * Phi_tcells;
    }
  }

  {
      Timer t_random(timers["randomize"]);
      // Unshuffle R (this updates also the class objects since this is a
      // reference to these class attributes)
      R_randomized = R_randomized.cols(reverse_index);
      dist_mat = dist_mat.cols(reverse_index);
  }
  return 0;
}


void harmony::moe_correct_ridge_cpp() {

  Z_corr = Z_orig;
  Progress p(K, verbose);

  VECTYPE sizes(sum(Phi, 1));
  VECTYPE _R;
  for (unsigned k = 0; k < K; k++) {

    p.increment();
    if (Progress::check_abort())
      return;
    
    VECTYPE avg_R(O.row(k).t() / sizes);
    bool subset_data = false;
    // Determine which batches do not belong to the cluster
    std::vector<unsigned> keep;
    std::vector<unsigned> keep_cols_scratch;
    std::vector<unsigned>cov_levels(B_vec.size(), 0);
    keep_cols_scratch.reserve(N*B_vec.size());
    // Estimate which covariates have sufficient support and need to
    // be corrected
    for (unsigned b = 0, current_covariate = 0; b < B; b++) {
      // Determine covariate of factor
      if ((current_covariate < covariate_bounds.size()) && !(b < covariate_bounds[current_covariate])) {
	current_covariate++;
      }
      
      float batch_representation  = as_scalar(avg_R.row(b));
      if (batch_representation > batch_proportion_cutoff) {
	// Increase the number of batches that qualify for correction
	cov_levels[current_covariate]++;
      }
    }
    
    // for (unsigned c =0; c < B_vec.size(); ++c ){
      
    //   std::cout << "Covariate level " << c << ": Included " << cov_levels[c] << " out of " << B_vec[c] << std::endl;
    // }

    // Collect the cells we need to correct
    for (unsigned b = 0, current_covariate = 0; b < B; b++) {
      
      // Increase covariate if we iterated through all the levels
      if ((current_covariate < covariate_bounds.size()) && !(b < covariate_bounds[current_covariate])) {	
	current_covariate++;
      }
      
      // Determine whether we need to include batch
      float batch_representation  = as_scalar(avg_R.row(b));
      if (batch_representation > batch_proportion_cutoff && cov_levels[current_covariate] > 1) {
	keep.push_back(b);
	keep_cols_scratch.insert(keep_cols_scratch.end(), index[b].memptr(), index[b].memptr() + index[b].n_rows);
      }
    }

    // Determine the active covariates pre-allocation of buffers
    unsigned active_covariates = 0;
    for (auto const& l: cov_levels) {
      if (l > 1) {
	active_covariates++;
      }
    }
    
    arma::uvec keep_batch = arma::conv_to<arma::uvec>::from(keep);
    MATTYPE *_Z_corr, *_Z_tmp;
    arma::uvec keep_cols;
    SPMAT *_Phi_moe, *_Phi_moe_t, *_lambda_mat, *__Rk;
    std::vector<arma::uvec>* _index;
    VECTYPE _R;
    unsigned PhiNonZero = 0;
    
    // Drop unused batches
    if (keep.size() == B) {
      // std::cout << "Normal pass of the data " << std::endl;
      // Avoid expensive copy, in this case we do not delete the
      // pointer as it is handled by the object lifetime
      _Z_corr = &(this->Z_corr);
      _Z_tmp = new MATTYPE(Z_orig);
      _Phi_moe = &(this->Phi_moe);
      _Phi_moe_t = &(this->Phi_moe_t);
      __Rk = new SPMAT(_Z_corr->n_cols, _Z_corr->n_cols);
      __Rk->diag() = this->R.row(k);
      _index = &(this->index);
      _lambda_mat = new SPMAT(B + 1, B + 1);
      
      if (!lambda_estimation) {
	// Set lambda
	_lambda_mat->diag() = lambda;
      } else {
	_lambda_mat->diag() = find_lambda_cpp(alpha, E.row(k).t());
      }
    } else {
      subset_data = true;
      // std::cout << "Filter " << B - keep.size() << " out of " << B << " batches" << std::endl;
      Timer t(timers["subset_overhead"]);
     

      //--Generate the new rowind, indptr. Determine the correct size
      // for the new sparse matrix. The size of entries in general are
      // N*number of covariates.
      if (active_covariates == 0) {
	// Rcpp::Rcout << "Skipping correction step" << std::endl;
	continue;
      }
      
      // Subset the old sparse design matrix and map to the new cell index
      
      std::set<unsigned> keep_cols_set(keep_cols_scratch.begin(), keep_cols_scratch.end());
      PhiNonZero = keep_cols_set.size() + keep_cols_scratch.size(); // Batches included + intercept
      // So we can slice easily arma data structures
      keep_cols = conv_to<arma::uvec>::from(std::vector<unsigned>(keep_cols_set.begin(),
								  keep_cols_set.end()));
      std::vector<int> cell_map(N, -1);
      {
	Timer t(timers["subset_overhead_mapping"]);
	unsigned i = 0;
	for (auto const& c: keep_cols_set) {	  
	  cell_map[c] = i++;
	}
      }
      
      // std::cout << "Total cells: " << keep_cols_set.size() << " in " <<  keep.size() << " batches" << std::endl;
      
      Timer* t1 = new Timer(timers["subset_prepare_timers"]);
      
      // Initialize the new sparse matrix buffers
      arma::uvec rowind_new(PhiNonZero); // Active covariates plus intercept
      arma::uvec indptr_new(keep.size() + 1 + 1); // +1 for the intercept, +1 for the indptr
      rowind_new.subvec(0, keep_cols_set.size() - 1) = linspace<uvec>(0, keep_cols_set.size() - 1, keep_cols_set.size());
      indptr_new[0] = 0;
      indptr_new[1] = keep_cols_set.size();

      // Get the new index
      _index = new std::vector<arma::uvec>();
      
      // Get data from existing data structure, values are 1 by default
      const uword* rowind_old = Phi_moe_t.row_indices;
      const uword* indptr_old = Phi_moe_t.col_ptrs;
      {
	Timer t(timers["subset_overhead_subsetting"]);
      
	for (unsigned i = 0; i < keep.size(); i++) {
	  unsigned cell_offset = 0; // for the indptr	
	  // Determine which covariate are is the level we are iterating
	  auto batch_id = keep[i];
	  unsigned max_idx = indptr_old[batch_id+2], min_idx = indptr_old[batch_id+1];
	  unsigned base_range = indptr_new(i+1);
	
	  for (unsigned idx = min_idx; idx < max_idx; ++idx) {
	    int new_index = cell_map[rowind_old[idx]];
	    rowind_new(base_range + (cell_offset++)) = new_index;
	  }
	  indptr_new(i+2) = base_range + cell_offset;	  
	  _index->push_back(rowind_new.subvec(base_range, indptr_new(i+2)-1));
	}
	
	delete t1;
      }
      
      {
	Timer t(timers["subset_overhead_buffers"]);
	{
	  Timer t(timers["subset_overhead_buffers_Z"]);
	  _Z_corr = new MATTYPE(this->Z_corr.cols(keep_cols));
	  _Z_tmp = new MATTYPE(this->Z_orig.cols(keep_cols));
	}
	{
	  Timer t(timers["subset_overhead_buffers_Phi"]);
	_Phi_moe_t = new SPMAT(rowind_new,
			       indptr_new,
			       VECTYPE(rowind_new.n_elem, arma::fill::ones),
			       keep_cols.n_elem,
			       keep.size() + 1);
	}
	{
	  Timer t(timers["subset_overhead_buffers_t()"]);
	  _Phi_moe = new SPMAT(_Phi_moe_t->t());
	}
	// Generate new Rs
	__Rk = new SPMAT(_Z_corr->n_cols, _Z_corr->n_cols);
	VECTYPE _R(this->R.row(k).as_col());

	__Rk->diag() = _R.rows(keep_cols);
	_lambda_mat = new SPMAT(keep.size() + 1, keep.size() + 1);
	if (!lambda_estimation) {
	  // Set lambda
	  VECTYPE _ltmp(keep_batch.n_rows + 1);
	  _ltmp(0) = 0;
	  _ltmp.subvec(1, keep_batch.n_rows) = lambda.rows(keep_batch+1);
	  _lambda_mat->diag() = _ltmp.as_row();
	} else {

	  VECTYPE Esub = E.row(k).as_col();
	  Esub = Esub.rows(keep_batch);
	  _lambda_mat->diag() = find_lambda_cpp(alpha, Esub).as_row();
	}
      }      
      
    }

    MATTYPE& Z_corr= (*_Z_corr);
    SPMAT& Phi_moe = (*_Phi_moe);
    SPMAT& Phi_moe_t = (*_Phi_moe_t);
    SPMAT& lambda_mat = (*_lambda_mat);
    SPMAT& _Rk = (*__Rk);
    MATTYPE& Z_tmp = (*_Z_tmp);
    std::vector<arma::uvec>& index = *_index;

    Timer t(timers["correct_ridge_loop"]);        

    
    Timer *t1 = new Timer(timers["Phi_Rk"]);
    SPMAT Phi_Rk = Phi_moe * _Rk;
    delete t1;
    
    // t1 = new Timer(timers["Phi_Rk_optimized"]);
    
    // // https://gitlab.com/conradsnicta/armadillo-code/-/blob/14.0.x/include/armadillo_bits/SpMat_bones.hpp?ref_type=heads#L69
    
    // VECTYPE Rk_vals(_Rk.values, _Rk.n_cols);
    // VECTYPE _Phi_Rk_vals(_Rk.n_cols * (active_covariates + 1));
    // const unsigned ac = active_covariates + 1;
    
    // unsigned n_cells = _Rk.n_cols;
    // // For each cell we have active_covariates + 1 
    // for (unsigned i1 = 0; i1 < n_cells; ++i1) {
    //   for (unsigned i2 = 0; i2 < ac; ++i2) {       
    // 	_Phi_Rk_vals((i1 * ac) + i2) = Rk_vals(i1);
    //   }
    // }
    // for (unsigned c = 0; c < active_covariates + 1; ++c) {
    //   _Phi_Rk_vals(arma::span(c * _Rk.n_cols, ((c + 1) * _Rk.n_cols) - 1)) =;
    // }
    
    // SPMAT Phi_Rk2(arma::uvec(Phi_moe.row_indices, Phi_moe.n_nonzero),
    // 		  arma::uvec(Phi_moe.col_ptrs, Phi_moe.n_cols + 1),
    // 		  _Phi_Rk_vals, Phi_moe.n_rows, Phi_moe.n_cols, false);
    
    // delete t1;
    

    MATTYPE inv_cov, Phi_cov;
    {
      Timer t(timers["Phi_cov"]);
      Phi_cov = Phi_Rk * Phi_moe_t + lambda_mat;
    }

    {
      Timer t(timers["arma_inv"]);
      inv_cov = arma::inv(Phi_cov);
    }

    // Calculate R-scaled PCs once
    {
      Timer t(timers["Z_tmp"]);
      Z_tmp = Z_tmp.each_row() % VECTYPE(_Rk.diag()).as_row();
    }

    // Generate the betas contribution of the intercept using the data
    // This erases whatever was written before in W
    {
      Timer t(timers["Z_intercept"]);
      W = inv_cov.unsafe_col(0) * sum(Z_tmp, 1).t();
    }

    // Calculate betas by calculating each batch contribution
    {
      Timer t(timers["batch_exprod"]);
      for (unsigned b=0; b < Phi_moe.n_rows - 1; b++) {
	// inv_conv is B+1xB+1 whereas index is B long
	W += inv_cov.unsafe_col(b+1) * sum(Z_tmp.cols(index[b]), 1).t();
      }
    }
    Y.col(k) = W.row(0).t(); // update centroids
    W.row(0).zeros(); // do not remove the intercept

    {
      Timer t(timers["update_Zcorr"]);
      Z_corr -= W.t() * Phi_Rk;
    }
    
    if (subset_data) {
      Timer t(timers["subset_overhead_assign"]);
      // Update original Zcorr columns
      this->Z_corr.cols(keep_cols) = Z_corr;
      // We don't need anything else
      delete _Z_corr;
      delete _Phi_moe;
      delete _Phi_moe_t;
      delete _index;
    }

    delete _lambda_mat;
    delete __Rk;
    delete _Z_tmp;

  }


  Y = arma::normalise(Y, 2, 0);

  
  print_timers();




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

RMAT harmony::getZcorr() {
  return conv_to<RMAT>::from(Z_corr.cols(original_index));
}

RMAT harmony::getZorig() {
  return conv_to<RMAT>::from(Z_orig.cols(original_index));
}


RCPP_MODULE(harmony_module) {
  class_<harmony>("harmony")
      .constructor()
      // .field("Z_corr", &harmony::Z_corr)
      // .field("Z_orig", &harmony::Z_orig)
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
    .method("getZcorr", &harmony::getZcorr)
    .method("getZorig", &harmony::getZorig)
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
