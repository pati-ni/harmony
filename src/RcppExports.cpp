// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include "harmony_types.h"
#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// kmeans_centers
MATTYPE kmeans_centers(const MATTYPE& X, const unsigned int K, bool verbose);
RcppExport SEXP _harmony_kmeans_centers(SEXP XSEXP, SEXP KSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const MATTYPE& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type K(KSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(kmeans_centers(X, K, verbose));
    return rcpp_result_gen;
END_RCPP
}
// scaleRows_dgc
MATTYPE scaleRows_dgc(const VECTYPE& x, const VECTYPE& p, const VECTYPE& i, int ncol, int nrow, float thresh);
RcppExport SEXP _harmony_scaleRows_dgc(SEXP xSEXP, SEXP pSEXP, SEXP iSEXP, SEXP ncolSEXP, SEXP nrowSEXP, SEXP threshSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const VECTYPE& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const VECTYPE& >::type p(pSEXP);
    Rcpp::traits::input_parameter< const VECTYPE& >::type i(iSEXP);
    Rcpp::traits::input_parameter< int >::type ncol(ncolSEXP);
    Rcpp::traits::input_parameter< int >::type nrow(nrowSEXP);
    Rcpp::traits::input_parameter< float >::type thresh(threshSEXP);
    rcpp_result_gen = Rcpp::wrap(scaleRows_dgc(x, p, i, ncol, nrow, thresh));
    return rcpp_result_gen;
END_RCPP
}
// find_lambda_cpp
VECTYPE find_lambda_cpp(const float alpha, const VECTYPE& cluster_E);
RcppExport SEXP _harmony_find_lambda_cpp(SEXP alphaSEXP, SEXP cluster_ESEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const float >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< const VECTYPE& >::type cluster_E(cluster_ESEXP);
    rcpp_result_gen = Rcpp::wrap(find_lambda_cpp(alpha, cluster_E));
    return rcpp_result_gen;
END_RCPP
}

RcppExport SEXP _rcpp_module_boot_harmony_module();

static const R_CallMethodDef CallEntries[] = {
    {"_harmony_kmeans_centers", (DL_FUNC) &_harmony_kmeans_centers, 3},
    {"_harmony_scaleRows_dgc", (DL_FUNC) &_harmony_scaleRows_dgc, 6},
    {"_harmony_find_lambda_cpp", (DL_FUNC) &_harmony_find_lambda_cpp, 2},
    {"_rcpp_module_boot_harmony_module", (DL_FUNC) &_rcpp_module_boot_harmony_module, 0},
    {NULL, NULL, 0}
};

RcppExport void R_init_harmony(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
