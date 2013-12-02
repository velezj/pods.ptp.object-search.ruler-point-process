

#define BOOST_TEST_MODULE autoscale-rejection-sampler
#include <boost/test/included/unit_test.hpp>

#include <iostream>
#include <ruler-point-process/ruler_point_process.hpp>
#include <point-process-experiment-core/experiment_utils.hpp>
#include <point-process-core/marked_grid.hpp>
#include "register-common-models.cpp"
#include <p2l-common/plots.hpp>
#include <point-process-core/histogram.hpp>
#include <sstream>

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
    pp = get_model_by_id( "test::model::ruler_2d_small_002", window ); 
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

BOOST_AUTO_TEST_SUITE( ruler_process )

//=========================================================================

BOOST_FIXTURE_TEST_CASE( stationary_samples_test, fixture_rp )
{
  size_t num_samples = 50;
  size_t skip_between_samples = 100;
  double grid_approximation_epsilon = 0.1;

  // setup the planner (and hecne the model) with *all* of the initial
  // grountruth data
  // setup_planner_with_initial_observations( planner,
  // 					   true,
  // 					   window,
  // 					   groundtruth );

  marked_grid_t<bool> groundtruth_grid 
    = point_set_as_grid( groundtruth, window, grid_approximation_epsilon );
  
  std::vector< std::vector<nd_point_t> > samples;
  std::vector< double > distance_to_groundtruth;
  for( size_t i = 0; i < num_samples; ++i ) {

    // draw a sample
    std::vector<nd_point_t> s
      = rpp->sample();
    samples.push_back( s );

    // compute the distance from the sample to the groundtruth 
    // using a marke-grid approximation for hte point set
    marked_grid_t<bool> sample_grid
      = point_set_as_grid( s, window, grid_approximation_epsilon );
    double dist = marked_grid_distance( sample_grid, groundtruth_grid );
    distance_to_groundtruth.push_back( dist );
    
    // skip some samples
    if( i+1 < num_samples ) {
      rpp->mcmc( skip_between_samples + 1 );
    }

    std::cout << "sampled [" << i << " / " << num_samples << "]: " << ( (double)i / num_samples * 100.0 ) << "%" << std::endl;
  }

  // make sure the distance at the beginning is more than at the end
  BOOST_CHECK_GT( distance_to_groundtruth[0],
		  distance_to_groundtruth[ distance_to_groundtruth.size()-1 ] );

  // print out hte distances
  std::cout << "d = [";
  for( auto d : distance_to_groundtruth ) {
    std::cout << d << ",";
  }
  std::cout << "];" << std::endl;

  std::vector<nd_point_t> distance_points;
  for( double d : distance_to_groundtruth ) {
    distance_points.push_back( point( d ) );
  }

  time_t ts = time(NULL);
  pid_t pid = getpid();
  std::ostringstream oss, label_ss;
  oss << "test-rp-distance-hist-prior-_" << ts << "_pid" << pid << ".svg";
  label_ss << num_samples << " | " << skip_between_samples << " | " << grid_approximation_epsilon << " | 100";
  svg_plot_histogram( oss.str(),
		      create_histogram( 100,
					distance_points ),
		      label_ss.str() );
					

  distance_to_groundtruth.insert( distance_to_groundtruth.begin(), 1.0 );
  oss.str("");
  label_ss.str("");
  oss << "test-rp-distance-line-prior_" << ts << "_pid" << pid << ".svg";
label_ss << num_samples << " | " << skip_between_samples << " | " << grid_approximation_epsilon;
  p2l::common::svg_plot( oss.str(),
			 distance_to_groundtruth,
			 label_ss.str() );

  
}

//=========================================================================
//=========================================================================

BOOST_AUTO_TEST_SUITE_END()

//=========================================================================
//=========================================================================
//=========================================================================
//=========================================================================
//=========================================================================
//=========================================================================
//=========================================================================
//=========================================================================
//=========================================================================
//=========================================================================
//=========================================================================
//=========================================================================
//=========================================================================
//=========================================================================
//=========================================================================
//=========================================================================
//=========================================================================
//=========================================================================
//=========================================================================
//=========================================================================
//=========================================================================
//=========================================================================
//=========================================================================
//=========================================================================
//=========================================================================
