
#if !defined( __P2L_RULER_POINT_PROCESS_tick_models_HPP__ )
#define __P2L_RULER_POINT_PROCESS_tick_models_HPP__

#include <math-core/polynomial.hpp>
#include <probability-core/uniform.hpp>
#include <iosfwd>

namespace ruler_point_process {


  //====================================================================
  
  // Description:
  // API for tick_model classes

  template<class TickModelT>
  struct global_parameters_of {
    typedef struct {} type;
  };
  
  
  template<class TickModelT>
  void
  flat_to_model( const std::vector<double>& flat,
		 TickModelT& model,
		 const typename global_parameters_of<TickModelT>::type& params);

  template<class TickModelT>
  std::vector<double>
  model_to_flat( const TickModelT& model,
		 const typename global_parameters_of<TickModelT>::type& params);

  template<class TickModelT>
  std::vector<math_core::nd_point_t>
  ticks_for_model( const TickModelT& model,
		   const typename global_parameters_of<TickModelT>::type& params );

  // Also needed:
  // probability_core::sample_from( uniform_distribution_t<TickModelT> )



  //====================================================================



  //======================================================================

  // Description:
  // Linear manifold with exactly 3 ticks
  struct linear_manifold_3_ticks {
    math_core::polynomial_t line;
    double length_scale;
  };


  template<>
  void
  flat_to_model( const std::vector<double>& flat,
		 linear_manifold_3_ticks& model,
		 const typename global_parameters_of<linear_manifold_3_ticks>::type& params)
  {
    assert( flat.size() == 3 );
    model.line = math_core::polynomial_t( { flat[0], flat[1] } );
    model.length_scale = flat[2];
  }

  template<>
  std::vector<double>
  model_to_flat( const linear_manifold_3_ticks& model,
		 const typename global_parameters_of<linear_manifold_3_ticks>::type& params)
  {
    std::vector<double> flat;
    flat.push_back( model.line.coefficients()[0] );
    flat.push_back( model.line.coefficients()[1] );
    flat.push_back( model.length_scale );
    return flat;
  }

  template<>
  std::vector<math_core::nd_point_t>
  ticks_for_model( const linear_manifold_3_ticks& model,
		   const typename global_parameters_of<linear_manifold_3_ticks>::type& params )
  {
    std::vector<math_core::nd_point_t> ticks;
    ticks.push_back( math_core::point( 0.0 * model.length_scale,
				       model.line.evaluate( 0.0 * model.length_scale ) ) );
    ticks.push_back( math_core::point( 1.0 * model.length_scale,
				       model.line.evaluate( 1 * model.length_scale ) ) );
    ticks.push_back( math_core::point( 2.0 * model.length_scale,
				       model.line.evaluate( 2 * model.length_scale ) ) );
    return ticks;
  }


  //======================================================================

  std::ostream& operator<< (std::ostream& os,
			    const linear_manifold_3_ticks& model )
  {
    os << model.line
       << " ls=" << model.length_scale;
    return os;
  }

  
  //======================================================================
  //======================================================================
  //======================================================================
  //======================================================================
  //======================================================================
  //======================================================================
  //======================================================================
  //======================================================================
  //======================================================================
  //======================================================================
  //======================================================================


}


namespace probability_core {

  template<>
  ruler_point_process::linear_manifold_3_ticks
  sample_from( const uniform_distribution_t<ruler_point_process::linear_manifold_3_ticks>& u )
  {
    // debug
    // std::cout << "sample(uniform<linear_manifold_3_ticks>): started" << std::endl;
    // std::cout << "  c=[" << u.support.first.line.coefficients()[0]
    // 	      << ", "
    // 	      << u.support.second.line.coefficients()[0]
    // 	      << "]" << std::endl;
    // std::cout << "  x=[" << u.support.first.line.coefficients()[1]
    // 	      << ", "
    // 	      << u.support.second.line.coefficients()[1]
    // 	      << "]" << std::endl;
    // std::cout << " s=[" << u.support.first.length_scale
    // 	      << ", "
    // 	      << u.support.second.length_scale
    // 	      << "]" << std::endl;
      
    
    
    ruler_point_process::linear_manifold_3_ticks model;
    double c, x, s;
    c = sample_from( uniform_distribution
		     ( u.support.first.line.coefficients()[0],
		       u.support.second.line.coefficients()[0] ) );
    x = sample_from( uniform_distribution
		     ( u.support.first.line.coefficients()[1],
		       u.support.second.line.coefficients()[1] ) );
    s = sample_from( uniform_distribution
		     ( u.support.first.length_scale,
		       u.support.second.length_scale ) );
    model.line = math_core::polynomial_t( {c,x} );
    model.length_scale = s;
    return model;
  }


}

#endif

