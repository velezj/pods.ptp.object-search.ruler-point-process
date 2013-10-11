
#if !defined( __MODEL_HPP__ )
#define __MODEL_HPP__

#include <lcmtypes/p2l_math_core.hpp>
#include <lcmtypes/p2l_probability_core.hpp>
#include <iosfwd>
#include <string>

namespace ruler_point_process {

  using namespace math_core;
  using namespace probability_core;

  
  // Description:
  // The parameters for the ruler point process
  struct ruler_point_process_model_t{

    double alpha;
    
    gamma_distribution_t precision_distribution;
    gamma_conjugate_prior_t period_distribution;
    gamma_conjugate_prior_t ruler_length_distribution;
    gaussian_distribution_t ruler_start_mean_distribution;
    gamma_distribution_t ruler_start_precision_distribution;
    gaussian_distribution_t ruler_direction_mean_distribution;
    gamma_distribution_t ruler_direction_precision_distribution;

    double prior_variance;

    double prior_ruler_period_p_shape;
    double prior_ruler_period_p_rate;
    double prior_ruler_period_q_shape;
    double prior_ruler_period_q_rate;
    double prior_ruler_period_r_shape;
    double prior_ruler_period_r_rate;
    double prior_ruler_period_s_shape;
    double prior_ruler_period_s_rate;

    double prior_ruler_length_p_shape;
    double prior_ruler_length_p_rate;
    double prior_ruler_length_q_shape;
    double prior_ruler_length_q_rate;
    double prior_ruler_length_r_shape;
    double prior_ruler_length_r_rate;
    double prior_ruler_length_s_shape;
    double prior_ruler_length_s_rate;

    // double prior_ruler_period_p_lambda;
    // double prior_ruler_period_q_lambda;
    // double prior_ruler_period_r_lambda;
    // double prior_ruler_period_s_lambda;
    // double prior_ruler_length_p_lambda;
    // double prior_ruler_length_q_lambda;
    // double prior_ruler_length_r_lambda;
    // double prior_ruler_length_s_lambda;


    math_core::nd_point_t prior_ruler_start_mean;
    double prior_ruler_start_variance;
    math_core::nd_point_t prior_ruler_direction_mean;
    double prior_ruler_direction_variance;
    
    
    ruler_point_process_model_t()
      : prior_variance( 1.0 ),

	prior_ruler_period_p_shape( 2 ),
	prior_ruler_period_p_rate( 0.5 ),
	prior_ruler_period_q_shape( 2 ),
	prior_ruler_period_q_rate( 0.5 ),
	prior_ruler_period_r_shape( 1 ),
	prior_ruler_period_r_rate( 0.5 ),
	prior_ruler_period_s_shape( 1 ),
	prior_ruler_period_s_rate( 0.5 ),


	prior_ruler_length_p_shape( 36 ),
	prior_ruler_length_p_rate( 1 ),
	prior_ruler_length_q_shape( 12 ),
	prior_ruler_length_q_rate( 1 ),
	prior_ruler_length_r_shape( 1 ),
	prior_ruler_length_r_rate( 0.5 ),
	prior_ruler_length_s_shape( 1 ),
	prior_ruler_length_s_rate( 0.5 ),

	// prior_ruler_period_p_lambda( 2 ),
	// prior_ruler_period_q_lambda( 2 ),
	// prior_ruler_period_r_lambda( 2 ),
	// prior_ruler_period_s_lambda( 2 ),
	// prior_ruler_length_p_lambda( 2 ),
	// prior_ruler_length_q_lambda( 2 ),
	// prior_ruler_length_r_lambda( 2 ),
	// prior_ruler_length_s_lambda( 2 ),

	prior_ruler_start_mean( ),
	prior_ruler_start_variance( 100 ),
	prior_ruler_direction_mean(),
	prior_ruler_direction_variance( 100 )
    {}

  };


  // Description:
  // The ruler point process state
  typedef struct {
    
    // the model parameters
    ruler_point_process_model_t model;
    
    // the observations
    std::vector<nd_point_t> observations;
    
    // the set of distributiuons for a mixture
    // we have the giassians around ticks
    // and period gammas
    // and ruler length gamms
    // and ruler start gaussians
    std::vector< gaussian_distribution_t > mixture_gaussians;
    std::vector< gamma_distribution_t > mixture_period_gammas;
    std::vector< gamma_distribution_t > mixture_ruler_length_gammas;
    std::vector< gaussian_distribution_t > mixture_ruler_start_gaussians;
    std::vector< gaussian_distribution_t > mixture_ruler_direction_gaussians;

    // the correspondence between observation and mixture lement
    std::vector< size_t > observation_to_mixture;

    // the mapping between a mixture element and it's observations (as indices)
    std::vector<std::vector< size_t > > mixture_to_observation_indices;

    // the negative observations
    std::vector<nd_aabox_t> negative_observations;

    // the window
    nd_aabox_t window;

    // do we want to trace the mcmc sampling
    bool trace_mcmc;
    std::string trace_mcmc_dir;

    // do we want to trace when we ask for samples from hte process
    bool trace_samples;
    std::string trace_samples_dir;

    // The iterations (of mcmc) done to this state
    double iteration;

  } ruler_point_process_state_t;


  // Description:
  // Returns the observation points for a partiuclar mixture in a state
  std::vector<nd_point_t> points_for_mixture( const ruler_point_process_state_t& state,
					      const size_t mixture_i );


  // Description:
  // Sample a point cloud from the igmm-point-process.
  std::vector<nd_point_t> sample_from( const ruler_point_process_state_t& state );


  // Description:
  // outputs a state (or model)
  std::ostream& operator<< (std::ostream& os,
			    const ruler_point_process_state_t& state );
  std::ostream& operator<< (std::ostream& os,
			    const ruler_point_process_model_t& model );




  // Description:
  // Prints out hte state in a single line
  void model_print_shallow_trace( const ruler_point_process_state_t& state,
				  std::ostream& out );
  void model_print_shallow_trace( const ruler_point_process_model_t& model,
				  std::ostream& out );
  


}

#endif

