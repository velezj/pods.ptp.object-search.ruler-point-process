
#define BOOST_TEST_MODULE ruler-point-process-plots
#include <boost/test/included/unit_test.hpp>

#include <iostream>
#include <ruler-point-process/ruler_point_process.hpp>
#include <point-process-experiment-core/experiment_utils.hpp>
#include "register-common-models.cpp"
#include <p2l-common/plots.hpp>
#include <point-process-core/histogram.hpp>
#include <sstream>
#include <ruler-point-process/plots.hpp>

#include <sys/types.h>
#include <unistd.h>
#include <time.h>


using namespace ruler_point_process;
using namespace point_process_experiment_core;
using namespace math_core;
using namespace point_process_core;


//=========================================================================



//=========================================================================

struct fixture_rp
{
  boost::shared_ptr<ruler_point_process_t> rpp;
  boost::shared_ptr<point_process_core::mcmc_point_process_t> pp;
  std::vector<nd_point_t> groundtruth;
  nd_aabox_t window;
  boost::shared_ptr<planner_core::grid_planner_t> planner;
  
  fixture_rp()
  {
    clear_all_registered_experiments();
    register_experiments();

    groundtruth = groundtruth_for_world( "test::world::2d_001" );
    window = window_for_world( "test::world::2d_001" );    
    pp = get_model_by_id( "test::model::ruler_2d_small_003", 
			  window, 
			  groundtruth ); 
    rpp = boost::dynamic_pointer_cast< ruler_point_process_t >( pp );
    planner = 
      get_planner_by_id( "test::planner::shortest_path_next_planner_001", pp );
  }
  ~fixture_rp()
  {
    clear_all_registered_experiments();
  }
};

//=========================================================================

BOOST_AUTO_TEST_SUITE( ruler_process_plots )

//=========================================================================

BOOST_FIXTURE_TEST_CASE( plots_test, fixture_rp )
{

  // step the point process for a bit
  size_t num_mcmc_steps = 20;
  rpp->set_liklihood_algorithm( ruler_point_process::mean_likelihood_approximation );
  size_t num_plots = 3;

  for( size_t p = 0; p < num_plots; ++p ) {
    for( size_t i = 0; i < num_mcmc_steps; ++i ) {
      rpp->single_mcmc_step();
      std::cout << ".";
      std::cout.flush();
    }
    std::cout << std::endl;

    
    std::ostringstream oss;
    oss << "test-rp-" << p;
    std::string plot_id =
      plot_ruler_point_process( *rpp.get(),
				oss.str() );
    std::cout << "PLOT-ID: " << plot_id << std::endl;
  }

}

//=========================================================================

BOOST_FIXTURE_TEST_CASE( plots_test_2, fixture_rp )
{

  // get a very particular point process
  int dim = 2;
  ruler_point_process_model_t model;
  model.prior_ruler_start_mean = zero_point( dim );
  model.prior_ruler_direction_mean = zero_point( dim );
  model.alpha = 200;
  model.precision_distribution.shape = 5000;
  model.precision_distribution.rate = 100;
  model.period_distribution.p = pow(4.0,10);
  model.period_distribution.q = 4.0*10;
  model.period_distribution.r = 10;
  model.period_distribution.s = 10;
  model.ruler_length_distribution.p = pow(10.0,10);
  model.ruler_length_distribution.q = 10.0 * 10;
  model.ruler_length_distribution.r = 10;
  model.ruler_length_distribution.s = 10;
  model.ruler_start_mean_distribution.dimension = dim;
  model.ruler_start_mean_distribution.means.push_back( 1.0 );
  model.ruler_start_mean_distribution.means.push_back( 1.0 );
  model.ruler_start_mean_distribution.covariance = to_dense_mat( Eigen::MatrixXd::Identity(dim,dim) * (2.0*2.0) );
  model.ruler_start_precision_distribution.shape = 50;
  model.ruler_start_precision_distribution.rate = 10;
  model.ruler_direction_mean_distribution.dimension = dim;
  model.ruler_direction_mean_distribution.means.push_back( 1.0 );
  model.ruler_direction_mean_distribution.means.push_back( 1.0 );
  model.ruler_direction_mean_distribution.covariance = to_dense_mat( Eigen::MatrixXd::Identity(dim,dim) * 1 );
  model.ruler_direction_precision_distribution.shape = 50;
  model.ruler_direction_precision_distribution.rate = 10;
  std::vector<nd_point_t> init_points;
  init_points.push_back( point( 2.7, 0.3 ) );
  init_points.push_back( point( 0.2, 0.2 ) );
  init_points.push_back( point( 1.2, 1.2 ) );
  init_points.push_back( point( 5.7, 3.3 ) );
  init_points.push_back( point( 2.2, 2.2 ) );
  init_points.push_back( point( 8.7, 6.3 ) );
  nd_aabox_t window = aabox( point( 0.0, 0.0 ),
			     point( 10.0, 10.0 ) );
  ruler_point_process_t rpp2( window,
			      model,
			      init_points );

  ruler_point_process_state_t state;
  state.window = window;
  state.model = model;
  state.trace_mcmc = false;
  state.trace_samples = false;
  state.iteration = 0;
  
  // create exactly the wanted mixutres
  state.observations.insert( state.observations.end(),
			     init_points.begin(),
			     init_points.end() );
  state.observation_to_mixture = { 0, 1, 1, 0, 1, 0 };
  state.mixture_to_observation_indices = 
    { { 0, 3, 5 },
      { 1, 2, 4 } };

  gaussian_distribution_t g1, g2;
  g1.dimension = dim;
  g1.means = { 0.0, 0.0 };
  g1.covariance.rows = g1.covariance.cols = 2;
  g1.covariance.num_elements = 4;
  g1.covariance.data = { 0.01, 0.0, 0.0, 0.01 };
  g2.dimension = dim;
  g2.means = { 0.0, 0.0 };
  g2.covariance.rows = g2.covariance.cols = 2;
  g2.covariance.num_elements = 4;
  g2.covariance.data = { 0.01, 0.0, 0.0, 0.01 };
  state.mixture_gaussians = { g1, g2 };

  gamma_distribution_t p1, p2;
  p1.shape = 2.0;
  p1.rate = 2.0;
  p2.shape = 2.0;
  p2.rate = 2.0 / 3.0;
  state.mixture_period_gammas = { p1, p2 };
  
  gamma_distribution_t l1, l2;
  l1.shape = 2.0;
  l1.rate = 2.0 / 3.0;
  l2.shape = 2.0;
  l2.rate = 2.0 / 8.0;
  state.mixture_ruler_length_gammas = { l1, l2 };

  gaussian_distribution_t s1, s2;
  s1.dimension = dim;
  s1.means = { 0.2, 0.2 };
  s1.covariance.rows = s1.covariance.cols = 2;
  s1.covariance.num_elements = 4;
  s1.covariance.data = { 0.002, 0.0, 0.0, 0.002 };
  s2.dimension = dim;
  s2.means = { 2.7, 0.3 };
  s2.covariance.rows = s2.covariance.cols = 2;
  s2.covariance.num_elements = 4;
  s2.covariance.data = { 0.002, 0.0, 0.0, 0.002 };
  state.mixture_ruler_start_gaussians = { s1, s2 };

  gaussian_distribution_t d1, d2;
  d1.dimension = dim;
  d1.means = { 1.0, 1.0 };
  d1.covariance.rows = d1.covariance.cols = 2;
  d1.covariance.num_elements = 4;
  d1.covariance.data = { 0.01, 0.0, 0.0, 0.01 };
  d2.dimension = dim;
  d2.means = { 1.0, 1.0 };
  d2.covariance.rows = d2.covariance.cols = 2;
  d2.covariance.num_elements = 4;
  d2.covariance.data = { 0.01, 0.0, 0.0, 0.01 };
  state.mixture_ruler_direction_gaussians = { d1, d2 };
  
  rpp2._state = state;
  
  size_t num_plots = 10;
  for( size_t p = 0; p < num_plots; ++p ) {
    std::ostringstream oss;
    oss << "test-rp-KNOWN_PERFECT-" << p;
    std::string plot_id =
      plot_ruler_point_process( rpp2,
				oss.str() );
    std::cout << "PLOT-ID: " << plot_id << std::endl;
    rpp2.mcmc( 30 );
  }
}


//=========================================================================

BOOST_FIXTURE_TEST_CASE( plots_test_3, fixture_rp )
{

  // get a very particular point process
  int dim = 2;
  ruler_point_process_model_t model;
  model.prior_ruler_start_mean = zero_point( dim );
  model.prior_ruler_direction_mean = zero_point( dim );
  model.alpha = 200;
  model.precision_distribution.shape = 5000;
  model.precision_distribution.rate = 100;
  model.period_distribution.p = pow(4.0,10);
  model.period_distribution.q = 4.0*10;
  model.period_distribution.r = 10;
  model.period_distribution.s = 10;
  model.ruler_length_distribution.p = pow(10.0,10);
  model.ruler_length_distribution.q = 10.0 * 10;
  model.ruler_length_distribution.r = 10;
  model.ruler_length_distribution.s = 10;
  model.ruler_start_mean_distribution.dimension = dim;
  model.ruler_start_mean_distribution.means.push_back( 1.0 );
  model.ruler_start_mean_distribution.means.push_back( 1.0 );
  model.ruler_start_mean_distribution.covariance = to_dense_mat( Eigen::MatrixXd::Identity(dim,dim) * (2.0*2.0) );
  model.ruler_start_precision_distribution.shape = 50;
  model.ruler_start_precision_distribution.rate = 10;
  model.ruler_direction_mean_distribution.dimension = dim;
  model.ruler_direction_mean_distribution.means.push_back( 1.0 );
  model.ruler_direction_mean_distribution.means.push_back( 1.0 );
  model.ruler_direction_mean_distribution.covariance = to_dense_mat( Eigen::MatrixXd::Identity(dim,dim) * 1 );
  model.ruler_direction_precision_distribution.shape = 50;
  model.ruler_direction_precision_distribution.rate = 10;
  std::vector<nd_point_t> init_points;
  init_points.push_back( point( 2.7, 0.3 ) );
  init_points.push_back( point( 0.2, 0.2 ) );
  init_points.push_back( point( 1.2, 1.2 ) );
  init_points.push_back( point( 5.7, 3.3 ) );
  init_points.push_back( point( 2.2, 2.2 ) );
  init_points.push_back( point( 8.7, 6.3 ) );
  nd_aabox_t window = aabox( point( 0.0, 0.0 ),
			     point( 10.0, 10.0 ) );
  ruler_point_process_t rpp2( window,
			      model,
			      init_points );

  ruler_point_process_state_t state;
  state.window = window;
  state.model = model;
  state.trace_mcmc = false;
  state.trace_samples = false;
  state.iteration = 0;
  
  // create exactly the wanted mixutres
  state.observations.insert( state.observations.end(),
			     init_points.begin(),
			     init_points.end() );
  state.observation_to_mixture = { 0, 1, 1, 0, 1, 0 };
  state.mixture_to_observation_indices = 
    { { 0, 3, 5 },
      { 1, 2, 4 } };

  gaussian_distribution_t g1, g2;
  g1.dimension = dim;
  g1.means = { 0.0, 0.0 };
  g1.covariance.rows = g1.covariance.cols = 2;
  g1.covariance.num_elements = 4;
  g1.covariance.data = { 0.01, 0.0, 0.0, 0.01 };
  g2.dimension = dim;
  g2.means = { 0.0, 0.0 };
  g2.covariance.rows = g2.covariance.cols = 2;
  g2.covariance.num_elements = 4;
  g2.covariance.data = { 0.01, 0.0, 0.0, 0.01 };
  state.mixture_gaussians = { g1, g2 };

  gamma_distribution_t p1, p2;
  p1.shape = 2.0;
  p1.rate = 2.0;
  p2.shape = 2.0;
  p2.rate = 2.0 / 3.0;
  state.mixture_period_gammas = { p1, p2 };
  
  gamma_distribution_t l1, l2;
  l1.shape = 2.0;
  l1.rate = 2.0 / 3.0;
  l2.shape = 2.0;
  l2.rate = 2.0 / 8.0;
  state.mixture_ruler_length_gammas = { l1, l2 };

  gaussian_distribution_t s1, s2;
  s1.dimension = dim;
  s1.means = { 0.2, 0.2 };
  s1.covariance.rows = s1.covariance.cols = 2;
  s1.covariance.num_elements = 4;
  s1.covariance.data = { 0.002, 0.0, 0.0, 0.002 };
  s2.dimension = dim;
  s2.means = { 2.7, 0.3 };
  s2.covariance.rows = s2.covariance.cols = 2;
  s2.covariance.num_elements = 4;
  s2.covariance.data = { 0.002, 0.0, 0.0, 0.002 };
  state.mixture_ruler_start_gaussians = { s1, s2 };

  gaussian_distribution_t d1, d2;
  d1.dimension = dim;
  d1.means = { 1.0, 1.0 };
  d1.covariance.rows = d1.covariance.cols = 2;
  d1.covariance.num_elements = 4;
  d1.covariance.data = { 0.01, 0.0, 0.0, 0.01 };
  d2.dimension = dim;
  d2.means = { 1.0, 1.0 };
  d2.covariance.rows = d2.covariance.cols = 2;
  d2.covariance.num_elements = 4;
  d2.covariance.data = { 0.01, 0.0, 0.0, 0.01 };
  state.mixture_ruler_direction_gaussians = { d1, d2 };
  
  rpp2._state = state;
  
  size_t num_plots = 3;
  for( size_t p = 0; p < num_plots; ++p ) {
    std::ostringstream oss;
    oss << "test-rp-KNOWN_PERFECT-" << p;
    std::string plot_id =
      plot_ruler_point_process( rpp2,
				oss.str() );
    std::cout << "PLOT-ID: " << plot_id << std::endl;
    rpp2.mcmc( 1 );
  }
}

//=========================================================================

BOOST_FIXTURE_TEST_CASE( plots_test_planner_1, fixture_rp )
{

  nd_aabox_t initial_window = aabox( point( 0.0, 0.0 ),
				     point( 7.0, 7.0 ) );
  nd_aabox_t true_init_window =
    setup_planner_with_initial_observations
    ( planner,
      false,
      initial_window,
      groundtruth );


  std::string pid = planner->plot( "test-planner-plot" );
  std::cout << "Planner PLOT-ID: " << pid << std::endl;
}

//=========================================================================


BOOST_AUTO_TEST_SUITE_END()
