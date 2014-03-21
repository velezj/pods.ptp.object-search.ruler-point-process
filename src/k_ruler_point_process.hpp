
#if !defined( __P2L_RULER_POINT_PROCESS_k_ruler_point_process_HPP__ )
#define __P2L_RULER_POINT_PROCESS_k_ruler_point_process_HPP__


#include <point-process-core/point_process.hpp>

namespace ruler_point_process {


  //=====================================================================

  //=====================================================================

  //=====================================================================

  // Description:
  // A ruler point process with a fixed (k) number of rulers
  template<class ManifoldModelT>
  class k_ruler_point_process_t
    : public point_process_core::mcmc_point_process_t
  {

  public:

    //-------------------------------------------------------------------

    // Description:
    // The paramete structure for this process.
    struct parameters_t {
      size_t k;
      model_prior_t prior;
    };

    //-------------------------------------------------------------------

    // Description:
    // The model structure for this process for an individual mixture.
    // This is a particular model and parameter setting
    struct model_t {
      math_core::nd_point_t start;
      size_t num_ticks;
      double length_scale;
      double spread;
      ManifoldModelT manifold;
    };

    //-------------------------------------------------------------------

    // Description:
    // A prior for a model_t for a single mixture
    struct model_prior_t {
      probability_core::gaussian_distribution_t start;
      probability_core::gamma_distribution_t num_ticks;
      probability_core::gamma_conjugate_prior_t length_scale;
      probability_core::gaussian_distribution_t spread;
      std::vector<probability_core::gaussian_distribution_t> coeffs;
    };

    //-------------------------------------------------------------------

    // Description:
    // A posterior for a model_t for a single mixture
    struct model_posterior_t {
      probability_core::gaussian_distribution_t start;
      probability_core::gamma_distribution_t num_ticks;
      probability_core::gamma_conjugate_prior_t length_scale;
      probability_core::gaussian_distribution_t spread;
      std::vector<probability_core::gaussian_distribution_t> coeffs;
    };

    //-------------------------------------------------------------------
    
  protected:

    // Description:
    // Parameters for this process
    parameters_t _params;

    // Description:
    // The window and observations so far
    math_core::nd_aabox_t _window;
    std::vector<math_core::nd_point_t> _observations;
    std::vector<math_core::nd_aabox_t> _negative_observations;
    
    // Description:
    // The mixture weights for the models, with prior and posterior
    std::vector<double> _mixture_weights;
    probability_core::dirichlet_distribution_t _mixture_weights_prior;
    probability_core::dirichlet_distribution_t _mixture_weights_posterior;
    
    // Description:
    // The current models, along with priors and posteriros
    std::vector<model_t> _models;
    std::vector<model_prior_t> _model_priors;
    std::vector<model_posterior_t> _model_posteriors;
 
  };
  

  //=====================================================================

}

#endif

