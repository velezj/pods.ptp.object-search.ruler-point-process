
#include <ruler-point-process/gem_k_ruler_process.hpp>
#include <math-core/io.hpp>
#include <iostream>

using namespace ruler_point_process;
using namespace math_core;
using namespace probability_core;

int main( int argc, char** argv )
{

  // create some data 
  std::vector<nd_point_t> data =
    { point( 0.1 ), point( 0.2 ), point( 0.3 ),
      point( 5.3 ), point( 6.0 ), point( 6.7 ) };

  // create some negative observations
  std::vector<nd_aabox_t> neg_obs =
    {  aabox( point( 0 ), point( 0.095 ) ),
       aabox( point( 0.105 ), point( 0.195 ) ),
       aabox( point( 0.205 ), point( 0.295 ) ),
       aabox( point( 0.305 ), point( 5.295 ) ),
       aabox( point( 5.305 ), point( 5.995 ) ),
       aabox( point( 6.005 ), point( 6.695 ) ),
       aabox( point( 6.705 ), point( 10.0 ) )
    };

  // some baseline rulerss and miture weights
  std::vector<ruler_t> base_rulers =
    { { point( 0.1 ), direction(vector(1)), 0.3, 0.1, 1e-5 },
      { point( 5.3 ), direction(vector(1)), 1.4, 0.7, 1e-5 } };
  std::vector<double> base_mixture_weights =
    { 0.5, 0.5 };

  // create the window
  nd_aabox_t window = aabox( point( 0 ), point( 10 ) );
  
  gem_k_ruler_process_parmaeters_t params;
  params.gem.max_optimize_iterations = 100;
  params.gem.stop.max_iterations = 100;
  params.num_rulers = 2;
  

  gem_k_ruler_process_t proc( window,
			      params,
			      data,
			      neg_obs );
  
  std::vector<ruler_t> rulers = proc.rulers();
  
  std::cout << "RULERS:" << std::endl;
  for( size_t i = 0; i < rulers.size(); ++i ) {
    ruler_t r = rulers[i];
    std::cout << "start: " << r.start << std::endl;
    std::cout << "     dir         : " << r.dir << std::endl;
    std::cout << "     length      : " << r.length << std::endl;
    std::cout << "     length-scale: " << r.length_scale << std::endl;
    std::cout << "     spread      : " << r.spread << std::endl;
  }

  std::cout << "found lik= " << proc.likelihood() << std::endl;
  proc._set_rulers( base_rulers );
  proc._set_ruler_mixture_weights( base_mixture_weights );
  std::cout << "base lik= " << proc.likelihood() << std::endl;

  return 0;
}
