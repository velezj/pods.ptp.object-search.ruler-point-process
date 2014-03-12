
#include <ruler-point-process/gem_k_strip_process.hpp>
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
      { point( 0.1, 0.1 ), point( 0.2, 0.2 ), point( 0.3, 0.3 ),
	point( 5.3, 5.3 ), point( 6.0, 6.0 ), point( 6.7, 6.7 ) };

    // create some negative observations
    // std::vector<nd_aabox_t> neg_obs =
    //   {  aabox( point( 0.3 + 0.1 ), point( 5.3 - 0.1 ) ),
    // 	 aabox( point( 5.3 + 0.1 ), point( 6.0 - 0.1 ) ),
    // 	 aabox( point( 6.0 + 0.1 ), point( 6.7 - 0.1 ) ),
    // 	 aabox( point( 6.7 + 0.1 ), point( 10.0 ) )
    //   };
    //std::vector<nd_aabox_t> neg_obs;
    std::vector<nd_aabox_t> neg_obs =
      {  aabox( point( 0.3 + 0.3,
		       0.3 + 0.3), 
		point( 5.3 - 0.3,
		       5.3 - 0.3) )
      };

    // some baseline rulerss and miture weights
    std::vector<strip_t> base_strips =
      { { point( 0.1, 0.1 ), polynomial_t( { 0.0, 1.0 } ), 0, 1.0, 0.1 },
	{ point( 5.3, 5.3 ), polynomial_t( { 0.0, 1.0 } ), 2, 0.7, 0.1 } };
    std::vector<double> base_mixture_weights =
      { 0.5, 0.5 };

    // create the window
    nd_aabox_t window = aabox( point( 0.0, 0.0 ), point( 10.0, 10.0 ) );
  
    gem_k_strip_process_parmaeters_t params;
    params.gem.max_optimize_iterations = 20;
    params.gem.stop.max_iterations = 500;
    params.gem.stop.relative_likelihood_tolerance = 1e-3;
    params.num_strips = 2;
    params.strip_poly_order = 2;
    params.num_gem_restarts = 10;
  

    gem_k_strip_process_t proc( window,
				params,
				data,
				neg_obs );
  
    std::vector<strip_t> rulers = proc.strips();
  
    std::cout << "MIXTURES: ";
    for(size_t i = 0; i < proc.mixture_weights().size(); ++i ) {
      std::cout << proc.mixture_weights()[i] << ", ";
    }
    std::cout << std::endl;
    std::cout << "STRIPS:" << std::endl;
    for( size_t i = 0; i < rulers.size(); ++i ) {
      strip_t r = rulers[i];
      std::cout << "start: " << r.start << std::endl;
      std::cout << "     manifold    : " << r.manifold << std::endl;
      std::cout << "     ticks       : " << r.num_ticks << std::endl;
      std::cout << "     length-scale: " << r.length_scale << std::endl;
      std::cout << "     spread      : " << r.spread << std::endl;
    }


    // print out some psamples
    std::cout << "samples: " << std::endl;
    size_t num_samples = 10;
    for( size_t i = 0; i < num_samples; ++i ) {
      std::vector<math_core::nd_point_t> s
	= proc.sample();
      std::cout << "  [" << i << "]#" << s.size() << ": ";
      for( math_core::nd_point_t p : s ) {
	std::cout << p << ", ";
      }
      std::cout << std::endl;
    }

    std::cout << "found lik= " << proc.likelihood() << std::endl;
    proc._set_strips( base_strips );
    proc._set_mixture_weights( base_mixture_weights );
    std::cout << "base lik= " << proc.likelihood() << std::endl;

  } 
  catch( boost::exception& e ) {
    std::cout << "EXCEPTION:" << std::endl;
    std::cout << boost::diagnostic_information(e);
  }

  return 0;
}
