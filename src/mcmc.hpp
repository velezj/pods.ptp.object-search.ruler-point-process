
#if !defined( __RULER_POINT_PROCESS_MCMC_HPP__ )
#define __RULER_POINT_PROCESS_MCMC_HPP__

#include "model.hpp"
#include <probability-core/distributions.hpp>


namespace ruler_point_process {

  using namespace math_core;
  using namespace probability_core;


  // Description:
  // A line
  struct line_model_t
  {
    double slope;
    double intercept;
  };
  

  // Description:
  // Runs mcmc for a given number of iterations.
  // This WILL change the given state
  void run_mcmc( ruler_point_process_state_t& state, 
		 const size_t num_iterations );


  // Description:
  // Perform a single step of mcmc.
  // This WILL change the given state
  void mcmc_single_step( ruler_point_process_state_t& state );
  

  // Description:
  // The likelihood for a single point for a mixture
  double likelihood_of_single_point_for_mixture
  ( const nd_point_t& point,
    const std::vector<nd_aabox_t>& negative_observations,
    const gaussian_distribution_t& spread_distribution,
    const gamma_distribution_t& period_distribution,
    const gamma_distribution_t& ruler_length_distribution,
    const gaussian_distribution_t& ruler_start_distribution,
    const gaussian_distribution_t& ruler_direction_distribution);

  
}

#endif

