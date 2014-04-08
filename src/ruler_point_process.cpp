
#include "ruler_point_process.hpp"
#include "plots.hpp"
#include <iostream>
#include <boost/math/special_functions/gamma.hpp>



namespace ruler_point_process {
  
  static double _dg_1 = boost::math::digamma( 1.0 );
  static double _dg_2 = boost::math::digamma( 2.0 );


  std::string
  ruler_point_process_t::plot( const std::string& title ) const
  {
    std::string pid = plot_ruler_point_process( *this, title );
    std::cout << "Ruler-Point-Process Plot: " << pid << std::endl;
    return pid;
  }
  
  
  double
  ruler_point_process_t::expected_entropy() const
  {
    return boost::math::digamma( this->_state.model.alpha + 1 ) - _dg_1;
  }

  double
  ruler_point_process_t::expected_posterior_entropy
  ( const std::vector<math_core::nd_point_t>& new_obs ) const
  {
    double N = new_obs.size();
    double a = this->_state.model.alpha;
    return boost::math::digamma( a + N + 1 ) 
      - ( ( a / (a + N) ) * _dg_1 )
      - ( ( N / (a + N) ) * _dg_2 );
  }

  double
  ruler_point_process_t::expected_posterior_entropy_difference
  ( const std::vector<math_core::nd_point_t>& new_obs ) const
  {
    double N = new_obs.size();
    double a = this->_state.model.alpha;
    double sum = 0;
    for( size_t k = 0; k < new_obs.size(); ++k ) {
      sum += ( 1.0 / ( a + 1 + k ) );
    }
    //std::cout << "  rp:ent-diff  " << ( ( ( a / (a + N) ) - 1.0 ) * _dg_1 ) << "  " << ( ( N / (a + N) ) * _dg_2 ) << "  - " << sum << std::endl;
    return fabs( ( ( ( a / (a + N) ) - 1.0 ) * _dg_1 )
		 + ( ( N / (a + N) ) * _dg_2 )
		 - sum );
  }

}
