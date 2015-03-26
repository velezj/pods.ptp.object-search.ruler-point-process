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
    //std::vector<nd_aabox_t> neg_obs;
     // std::vector<nd_aabox_t> neg_obs =
     //   {  aabox( point( 0.0, -3.0 ), 
     // 		 point( 2.0, 0.0 ) ) };
    // std::vector<nd_aabox_t> neg_obs =
    //   {  aabox( point( 0.0, -3.0 ), 
    // 		point( 2.0, 0.0 ) ),
    // 	 aabox( point( 2.0, 0.0 ), 
    // 		point( 5.0, 2.0 ) )
    //   };
    std::vector<nd_aabox_t> neg_obs =
      {  aabox( point( 0.1, -2.9 ), 
    		point( 1.9, -0.1 ) ),
    	 aabox( point( 2.1, 0.1 ), 
    		point( 4.9, 1.9 ) ),
	 aabox( point( 0.0, -4.5),
		point( 1.5, -3.0) )
      };

    
    // create the tick process parameters
    gem_k_tick_process_parameters_t<linear_manifold_3_ticks> params;
    params.gem.max_optimize_iterations = 100000;
    params.gem.stop.max_iterations = 500000;
    params.gem.stop.relative_likelihood_tolerance = 1e-1;
    params.k = 2;
    params.num_gem_restarts = 1;
    params.point_dimension = 2;
    params.tick_spread = 0.05;
    params.min_likelihood_epsilon = 1e-7;
    params.tick_model_flat_lower_bounds = { {-5.2, 0.8, 0.8},
					    {-5.2, 0.8, 0.8} };
    params.tick_model_flat_upper_bounds = { {0.2, 1.2 , 2.2},
					    {0.2, 1.2 , 2.2} };

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
      std::cout << "    ticks: ";
      std::vector<nd_point_t> ticks = ticks_for_model( models[i], params.tick_model_parameters );
      for( nd_point_t tick : ticks ) {
	std::cout << tick << " , ";
      }
      std::cout << std::endl;
    }

    // ok, check the likelihood of the "true" parameters
    std::vector<linear_manifold_3_ticks> true_models =
      { { polynomial_t({ 0.0, 1.0 }) , 1.0 },
	{ polynomial_t({ -5.0, 1.0}) , 2.0 } };
    std::vector<double> true_weights =
      { 0.5, 0.5 };
    process._set_tick_models( true_models );
    process._set_mixture_weights( true_weights );
    double true_lik = process.likelihood();
    std::cout << "lik(true model) = " << true_lik << std::endl;
    weights = process.mixture_weights();
    models = process.tick_models();
    for( size_t i = 0; i < weights.size(); ++i ) {
      std::cout << "  (True) Mix " << i << ":"
		<< " (True) w= " << weights[i]
		<< " (True) model= " << models[i] << std::endl;
      std::cout << "    (True) ticks: ";
      std::vector<nd_point_t> ticks = ticks_for_model( models[i], params.tick_model_parameters );
      for( nd_point_t tick : ticks ) {
	std::cout << tick << " , ";
      }
      std::cout << std::endl;
    }

      
  }
  catch( boost::exception& e ) {
    std::cout << "EXCEPTION:" << std::endl;
    std::cout << boost::diagnostic_information(e);
  }

  return 0;
}

