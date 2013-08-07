
#include <ruler-point-process/ruler_point_process.hpp>
#include <point-process-experiment-core/simulated_data.hpp>
#include <math-core/io.hpp>
#include <math-core/geom.hpp>
#include <math-core/matrix.hpp>
#include <probability-core/distribution_utils.hpp>
#include <iostream>
#include <fstream>
#include <sstream>


using namespace ruler_point_process;
using namespace math_core;
using namespace probability_core;
using namespace point_process_experiment_core;


int main( int argc, char** argv )
{

  // get the directory name from command line args if there
  std::string dir = "./";
  if( argc > 1 ) {
    dir = argv[1];
  }

  // log file of traces
  std::ostringstream oss_trace;
  oss_trace << dir << "/" << "mcmc_trace.data";
  std::ostringstream oss_meta;
  oss_meta << dir << "/" << "experiment.meta";
  std::ofstream fout_trace( oss_trace.str() );
  std::ofstream fout_meta( oss_meta.str() );

  // create a set of poitns (1d)
  size_t dim = 1; 
  nd_aabox_t window;
  window.n = dim;
  window.start = point( 0 );
  window.end = point( 20 );
  // std::vector<nd_point_t> points
  //   = simulate_line_point_clusters_gaussian_spread_poisson_size
  //   ( window,
  //     4,
  //     0.2,
  //     1 );
  std::vector<nd_point_t> points;
  points.push_back( point( 1.0 ) );
  //points.push_back( point( 2.0 ) );
  points.push_back( point( 3.0 ) );
  //points.push_back( point( 4.0 ) );
  points.push_back( point( 5.0 ) );
  // points.push_back( point( 7.0 ) );
  points.push_back( point( 8.5 ) );
  points.push_back( point( 9.5 ) );
  points.push_back( point( 10.5 ) );
  points.push_back( point( 11.5 ) );
  points.push_back( point( 12.5 ) );
  // points.push_back( point( 1.25 ) );
  // points.push_back( point( 3.25 ) );
  // points.push_back( point( 5.25 ) );

  std::vector<nd_aabox_t> neg_obs;
  neg_obs.push_back( aabox( point( 0 ), point( 0.995 ) ) );
  // neg_obs.push_back( aabox( point( 1.05 ), point( 1.20 ) ) );
  // neg_obs.push_back( aabox( point( 1.30 ), point( 2.95 ) ) );
  // neg_obs.push_back( aabox( point( 3.05 ), point( 3.20 ) ) );
  // neg_obs.push_back( aabox( point( 3.30 ), point( 4.95 ) ) );
  // neg_obs.push_back( aabox( point( 5.05 ), point( 5.20 ) ) );
  neg_obs.push_back( aabox( point( 1.005 ), point( 2.995 ) ) );
  neg_obs.push_back( aabox( point( 3.005 ), point( 4.995 ) ) );
  neg_obs.push_back( aabox( point( 5.005 ), point( 8.495 ) ) );
  neg_obs.push_back( aabox( point( 8.505 ), point( 9.495 ) ) );
  neg_obs.push_back( aabox( point( 9.505 ), point( 10.495 ) ) );
  neg_obs.push_back( aabox( point( 10.505 ), point( 11.495 ) ) );
  neg_obs.push_back( aabox( point( 11.505 ), point( 12.495 ) ) );

  // window.end = point( 100 );
  // // std::vector<nd_point_t> points
  // //   = simulate_line_point_clusters_gaussian_spread_poisson_size
  // //   ( window,
  // //     4,
  // //     0.2,
  // //     1 );
  // std::vector<nd_point_t> points;
  // points.push_back( point( 10 ) );
  // //points.push_back( point( 20 ) );
  // points.push_back( point( 30 ) );
  // //points.push_back( point( 40 ) );
  // points.push_back( point( 50 ) );
  // points.push_back( point( 70 ) );
  // points.push_back( point( 81 ) );
  // points.push_back( point( 82 ) );
  // points.push_back( point( 83 ) );

  

  // create a model and point process
  ruler_point_process_model_t model;
  model.prior_ruler_start_mean = zero_point( dim );
  model.prior_ruler_direction_mean = zero_point( dim );
  model.alpha = 1;
  model.precision_distribution.shape = 500000;
  model.precision_distribution.rate = 1000;
  model.period_distribution.p = 1;
  model.period_distribution.q = 1;
  model.period_distribution.r = 1;
  model.period_distribution.s = 1;
  model.ruler_length_distribution.p = 4*4*4*4;
  model.ruler_length_distribution.q = 4+4+4+4;
  model.ruler_length_distribution.r = 4;
  model.ruler_length_distribution.s = 4;
  model.ruler_start_mean_distribution.dimension = dim;
  model.ruler_start_mean_distribution.means.push_back( 1 );
  model.ruler_start_mean_distribution.covariance = to_dense_mat( Eigen::MatrixXd::Identity(dim,dim) * 10 );
  model.ruler_start_precision_distribution.shape = 5000;
  model.ruler_start_precision_distribution.rate = 100;
  model.ruler_direction_mean_distribution.dimension = dim;
  model.ruler_direction_mean_distribution.means.push_back( 10 );
  model.ruler_direction_mean_distribution.covariance = to_dense_mat( Eigen::MatrixXd::Identity(dim,dim) * 1 );
  model.ruler_direction_precision_distribution.shape = 5000;
  model.ruler_direction_precision_distribution.rate = 100;
  // boost::shared_ptr<ruler_point_process_t> process = 
  //   boost::shared_ptr<ruler_point_process_t>
  //   ( new ruler_point_process_t( window,
  // 				 model,
  // 				 points ) );

  // setup the process to have the "correct" mixture initially
  // to see if the mcmc sampling stays at this location
  std::vector<nd_point_t> points_0;
  points_0.push_back( points[0] );
  points_0.push_back( points[1] );
  points_0.push_back( points[2] );
  std::vector<nd_point_t> points_1;
  points_1.push_back( points[3] );
  points_1.push_back( points[4] );
  points_1.push_back( points[5] );
  points_1.push_back( points[6] );
  points_1.push_back( points[7] );
  boost::shared_ptr<ruler_point_process_t> process = 
    boost::shared_ptr<ruler_point_process_t>
    ( new ruler_point_process_t( window,
  				 model,
  				 points_0 ) );
  process->add_observations( points_1 );

  // add some negative observations
  for( int i = 0; i < neg_obs.size(); ++i ) {
    process->add_negative_observation( neg_obs[i] );
  }


  // ok, writeo ut hte experiment details to the meta file
  fout_meta << "# OBSERVATIONS" << std::endl;
  fout_meta << process->_state.observations.size() << std::endl;
  for( int i = 0; i < process->_state.observations.size(); ++i ) {
    fout_meta << process->_state.observations[i] << std::endl;
  }
  fout_meta << "# NEGATIVE OBSERVATION REGIONS" << std::endl;
  fout_meta << process->_state.negative_observations.size() << std::endl;
  for( int i = 0; i < process->_state.negative_observations.size(); ++i ) {
    fout_meta << process->_state.negative_observations[i] << std::endl;
  }
  fout_meta << "# INIT MIXTURE CLUSTERINGS" << std::endl;
  for( int i = 0; i < process->_state.observations.size(); ++i ) {
    fout_meta << process->_state.observation_to_mixture[i] << std::endl;
  }
  fout_meta << "# INIT MODEL" << std::endl;
  fout_meta << process->_state.model << std::endl;
  fout_meta << "# INIT MIXTURES" << std::endl;
  fout_meta << process->_state.mixture_gaussians.size() << std::endl;
  for( int i = 0; i < process->_state.mixture_gaussians.size(); ++i ) {
    fout_meta << "## MIXUTRE " << i << std::endl;
    fout_meta << "### SPREAD" << std::endl;
    fout_meta << process->_state.mixture_gaussians[i] << std::endl;
    fout_meta << "### PERIOD" << std::endl;
    fout_meta << process->_state.mixture_period_gammas[i] << std::endl;
    fout_meta << "### LENGTH" << std::endl;
    fout_meta << process->_state.mixture_ruler_length_gammas[i] << std::endl;
    fout_meta << "### START" << std::endl;
    fout_meta << process->_state.mixture_ruler_start_gaussians[i] << std::endl;
    fout_meta << "### DIRECTION" << std::endl;
    fout_meta << process->_state.mixture_ruler_direction_gaussians[i] << std::endl;
  }
  fout_meta << "# WINDOW" << std::endl;
  fout_meta << process->_state.window << std::endl;
  fout_meta << "# PRIORS" << std::endl;
  fout_meta << process->_state.model.prior_variance << std::endl;
  fout_meta << process->_state.model.prior_ruler_period_p_shape << std::endl;
  fout_meta << process->_state.model.prior_ruler_period_p_rate << std::endl;
  fout_meta << process->_state.model.prior_ruler_period_q_shape << std::endl;
  fout_meta << process->_state.model.prior_ruler_period_q_rate << std::endl;
  fout_meta << process->_state.model.prior_ruler_period_r_shape << std::endl;
  fout_meta << process->_state.model.prior_ruler_period_r_rate << std::endl;
  fout_meta << process->_state.model.prior_ruler_period_s_shape << std::endl;
  fout_meta << process->_state.model.prior_ruler_period_s_rate << std::endl;
  fout_meta << process->_state.model.prior_ruler_length_p_shape << std::endl;
  fout_meta << process->_state.model.prior_ruler_length_p_rate << std::endl;
  fout_meta << process->_state.model.prior_ruler_length_q_shape << std::endl;
  fout_meta << process->_state.model.prior_ruler_length_q_rate << std::endl;
  fout_meta << process->_state.model.prior_ruler_length_r_shape << std::endl;
  fout_meta << process->_state.model.prior_ruler_length_r_rate << std::endl;
  fout_meta << process->_state.model.prior_ruler_length_s_shape << std::endl;
  fout_meta << process->_state.model.prior_ruler_length_s_rate << std::endl;
  fout_meta << process->_state.model.prior_ruler_start_mean << std::endl;
  fout_meta << process->_state.model.prior_ruler_start_variance << std::endl;
  fout_meta << process->_state.model.prior_ruler_direction_mean << std::endl;
  fout_meta << process->_state.model.prior_ruler_direction_variance << std::endl;

  
  // do some plaing
  long num_samples = 100000;
  for( size_t i = 0; i < num_samples; ++i ) {
    
    mcmc_single_step( process->_state );

    if( i % (num_samples/10) == 0 ) {
      std::cout << "[" << i << "]--------------------------------------------" << std::endl;
      std::cout << process->_state << std::endl;
      std::cout << "--------------------------------------------" << std::endl;
    }

    // save the traces matrix row
    fout_trace 
      << i << " "
      << process->_state.model.alpha << " "
      << process->_state.model.precision_distribution.shape << " "
      << process->_state.model.precision_distribution.rate << " "
      << process->_state.model.period_distribution.p << " "
      << process->_state.model.period_distribution.q << " "
      << process->_state.model.period_distribution.r << " "
      << process->_state.model.period_distribution.s << " ";
    double m, var;
    estimate_gamma_conjugate_prior_sample_stats
      ( process->_state.model.period_distribution, m, var );
    fout_trace
      << m << " " 
      << var << " "
      << process->_state.model.ruler_length_distribution.p << " "
      << process->_state.model.ruler_length_distribution.q << " "
      << process->_state.model.ruler_length_distribution.r << " "
      << process->_state.model.ruler_length_distribution.s << " ";
    estimate_gamma_conjugate_prior_sample_stats
      ( process->_state.model.ruler_length_distribution, m, var );
    fout_trace
      << m << " "
      << var << " "
      << process->_state.model.ruler_start_mean_distribution.means[0] << " "
      << process->_state.model.ruler_start_mean_distribution.covariance.data[0] << " "
      << process->_state.model.ruler_start_precision_distribution.shape << " "
      << process->_state.model.ruler_start_precision_distribution.rate << " "
      << process->_state.model.ruler_direction_mean_distribution.means[0] << " "
      << process->_state.model.ruler_direction_mean_distribution.covariance.data[0] << " "
      << process->_state.model.ruler_direction_precision_distribution.shape << " "
      << process->_state.model.ruler_direction_precision_distribution.rate << " "
      << process->_state.observations.size() << " ";
    for( int i = 0; i < process->_state.observation_to_mixture.size(); ++i ) {
      fout_trace << process->_state.observation_to_mixture[i] << " ";
    }
    fout_trace << process->_state.mixture_gaussians.size() << " ";
    for( int i = 0; i < process->_state.mixture_gaussians.size(); ++i ) {
      fout_trace << process->_state.mixture_gaussians[i].means[0] << " "
		 << process->_state.mixture_gaussians[i].covariance.data[0] << " "
		 << process->_state.mixture_period_gammas[i].shape << " " 
		 << process->_state.mixture_period_gammas[i].rate << " "
		 << mean(process->_state.mixture_period_gammas[i]) << " "
		 << variance( process->_state.mixture_period_gammas[i]) << " "
		 << process->_state.mixture_ruler_length_gammas[i].shape << " "
		 << process->_state.mixture_ruler_length_gammas[i].rate << " "
		 << mean( process->_state.mixture_ruler_length_gammas[i] ) << " "
		 << variance( process->_state.mixture_ruler_length_gammas[i] ) << " "
		 << process->_state.mixture_ruler_start_gaussians[i].means[0] << " "
		 << process->_state.mixture_ruler_start_gaussians[i].covariance.data[0] << " "
		 << process->_state.mixture_ruler_direction_gaussians[i].means[0] << " "
		 << process->_state.mixture_ruler_direction_gaussians[i].covariance.data[0] << " ";
    }
    fout_trace << std::endl;
    fout_trace.flush();
      
  }

  return 0;
}
