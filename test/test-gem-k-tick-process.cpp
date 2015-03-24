#include <ruler-point-process/gem_k_tick_process.hpp>
#include <math-core/io.hpp>
#include <iostream>

using namespace ruler_point_process;
using namespace math_core;
using namespace probability_core;



int main( int argc, char** argv )
{

  // print out info on any exceptions
  try {


    // create some data 
    std::vector<nd_point_t> data =
      { point( 0.0, 0.0 ), point( 1.0, 1.0 ), point( 2.0, 2.0 ),
	point( 0.0, -5.0 ), point( 2.0, -3.0 ), point( 4.0, -1.0 ) };
    
    // create some negative observations
    std::vector<nd_aabox_t> neg_obs =
      {  aabox( point( 0.0, -3.0 ), 
		point( 2.0, 0.0 ) ),
    	 aabox( point( 2.0, 0.0 ), 
    		point( 5.0, 2.0 ) )
      };
    
    // create the tick process parameters
    gem_k_tick_process_parameters_t<linear_manifold_3_ticks> params;
    params.gem.max_optimize_iterations = 1000;
    params.gem.stop.max_iterations = 5000;
    params.gem.stop.relative_likelihood_tolerance = 1e-3;
    params.k = 2;
    params.num_gem_restarts = 10;
    params.point_dimension = 2;
    params.tick_spread = 0.05;
    params.min_likelihood_epsilon = 1e-4;
    params.tick_model_flat_lower_bounds = { {-10.0, -1.0, 0.001},
					    {-10.0, -1.0, 0.001} };
    params.tick_model_flat_upper_bounds = { {10.0, 1.0 , 10.0},
					    {10.0, 1.0 , 10.0} };

    // ok, create teh window
    nd_aabox_t window = aabox( point( -10.0, -10.0 ), point( 10.0, 10.0 ) );

    // debug
    std::cout << "About the create the gem-k-tick process object...." << std::endl;

    // now craete the tick process
    gem_k_tick_process_t<linear_manifold_3_ticks>
      process( window,
	       params,
	       data,
	       neg_obs);


    // print out the mixture and models
    std::cout << "Mixtures: " << std::endl;
    std::vector<double> weights = process.mixture_weights();
    std::vector<linear_manifold_3_ticks> models = process.tick_models();
    for( size_t i = 0; i < weights.size(); ++i ) {
      std::cout << "  Mix " << i << ":"
		<< " w= " << weights[i]
		<< " model= " << models[i] << std::endl;
    }
    
  }
  catch( boost::exception& e ) {
    std::cout << "EXCEPTION:" << std::endl;
    std::cout << boost::diagnostic_information(e);
  }

  return 0;
}

