
#include "gem_k_ruler_process.hpp"
#include <probability-core/distribution_utils.hpp>
#include <math-core/geom.hpp>
#include <math-core/matrix.hpp>
//#include <boost/exception.hpp>
#include <stdexcept>

using namespace math_core;
using namespace probability_core;

namespace ruler_point_process {

  //====================================================================

  std::vector<double> 
  gem_k_ruler_process_t::to_flat_vector( const ruler_t& r )
  {
    std::vector<double> res;
    res.insert(res.end(), 
	       r.start.coordinate.begin(),
	       r.start.coordinate.end());
    res.insert(res.end(),
	       r.dir.value.begin(),
	       r.dir.value.end() );
    res.push_back( r.length );
    res.push_back( r.length_scale );
    res.push_back( r.spread );
    return res;
  }

  //====================================================================
  
  ruler_t 
  gem_k_ruler_process_t::to_ruler( const std::vector<double>& v,
				   const size_t ndim )
  {
    assert( v.size() >= (2*ndim+2 + 1) );
    if( v.size() < (2 * ndim + 2 + 1) ) {
      //BOOST_THROW_EXCEPTION( flatten_error() );
      throw std::length_error("cannot unpack flat vector to ruler: to few elements!");
    }
    ruler_t r;
    r.start = point( ndim, v.data() );
    r.dir.n = ndim;
    r.dir.value.insert( r.dir.value.end(),
			v.begin() + ndim,
			v.begin() + 2 * ndim );
    r.length = v[2*ndim];
    r.length_scale = v[2*ndim+1];
    r.spread = v[2*ndim+2];
    return r;
  }


  //====================================================================

  std::vector<ruler_t>
  gem_k_ruler_process_t::create_initial_rulers() const
  {
    std::vector<ruler_t> rulers;
    for( size_t i = 0; i < _params.num_rulers; ++i ) {
      ruler_t r;
      r.start = sample_from( uniform_distribution( _window.start,
						   _window.end ) );
      nd_point_t dp = sample_from( uniform_distribution( _window.start,
							 _window.end ) );
      r.dir = direction( dp - centroid(_window) );
      double wdist = distance( _window.start,
			       _window.end );
      r.length = sample_from( uniform_distribution( 0.1 * wdist,
						    0.5 * wdist ) );
      r.length_scale = sample_from( uniform_distribution( 0.1 * r.length,
							  0.5 * r.length) );
      r.spread = sample_from( uniform_distribution( 0.1 * r.length_scale,
						    0.5 * r.length_scale ) );
      rulers.push_back( r );
    }
    return rulers;
  }

  //====================================================================


  void gem_k_ruler_process_t::_run_GEM()
  {
    if( _rulers.size() != _params.num_rulers ) {
      _rulers = create_initial_rulers();
    }
    
    // convert from rulers to double-vectors for GEM
    std::vector<std::vector<double> > ruler_params;
    for( size_t i = 0; i < _rulers.size(); ++i ) {
      ruler_params.push_back( to_flat_vector( _rulers[i] ) );
    }

    // the resulting parameter vector
    std::vector<std::vector<double> > mle_estimate;
    std::vector<double> mle_mixtures;

    // the likeliohood function
    using std::placeholders::_1;
    using std::placeholders::_2;
    std::function<double(const math_core::nd_point_t& single_data,
			 const std::vector<double>& params)>
      lik_f = std::bind( &gem_k_ruler_process_t::lik_single_point_single_ruler_flat,
			 *this,
			 _1,
			 _2);
    
    // run GEM
    run_GEM_mixture_model_MLE_numerical
      ( _params.gem,
	_observations,
	ruler_params,
	lik_f,
	mle_estimate,
	mle_mixtures );

    // convert doubles into rulers
    for( size_t i = 0; i < _rulers.size(); ++i ) {
      _rulers[i] = to_ruler( mle_estimate[i], _ndim );
    }
  }

  //====================================================================

  std::vector<nd_point_t> ticks_for_ruler( const ruler_t& r )
  {
    std::vector<nd_point_t> ticks;
    ticks.push_back( r.start );
    while( distance( ticks[ticks.size()-1], r.start ) < r.length ) {
      nd_point_t t = ticks[ ticks.size() - 1] + r.length_scale * r.dir; 
      ticks.push_back( t );
    }
    return ticks;
  }

  //====================================================================

  double gem_k_ruler_process_t::lik_single_point_single_ruler
  ( const math_core::nd_point_t& x,
    const ruler_t& ruler ) const
  {
    double p = 0;
    gaussian_distribution_t gauss;
    gauss.dimension = 1;
    gauss.means = { 0.0 };
    gauss.covariance = diagonal_matrix( point( ruler.spread ) );
    std::vector<nd_point_t> ticks = ticks_for_ruler(ruler);
    for( size_t i = 0; i < ticks.size(); ++i ) {
      double dist = distance(ticks[i],x);
      return pdf( point(dist), gauss );
    }
  }

  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================


}
