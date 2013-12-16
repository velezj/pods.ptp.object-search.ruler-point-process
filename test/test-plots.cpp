
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
    pp = get_model_by_id( "test::model::ruler_2d_small_003", window ); 
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

  std::string plot_id =
    plot_ruler_point_process( *rpp.get(),
			      "test-rpp" );
  std::cout << "PLOT-ID: " << plot_id << std::endl;

}


BOOST_AUTO_TEST_SUITE_END()
