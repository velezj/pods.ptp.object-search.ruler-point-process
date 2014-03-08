
#include "gem_k_ruler_process.hpp"
#include <probability-core/distribution_utils.hpp>
#include <math-core/geom.hpp>
#include <math-core/matrix.hpp>
#include <math-core/io.hpp>
#include <math-core/mpt.hpp>
//#include <boost/exception.hpp>
#include <stdexcept>
#include <iostream>
#include <cmath>

using namespace math_core;
using namespace probability_core;

namespace ruler_point_process {

  //====================================================================

  // Description:
  // encodes a point into a flat version of itself
  math_core::nd_point_t encode_point( const math_core::nd_point_t& p )
  {
    math_core::nd_point_t res;
    res.n = 2 * p.n + 1;
    res.coordinate.push_back( 1.0 );
    res.coordinate.insert( res.coordinate.end(),
			   p.coordinate.begin(),
			   p.coordinate.end() );
    for( size_t i = 0; i < p.n; ++i ) {
      res.coordinate.push_back( std::numeric_limits<double>::quiet_NaN() );
    }
    return res;
  }

  // description:
  // decodes a point from a falt point
  math_core::nd_point_t decode_point( const math_core::nd_point_t& flat )
  {
    assert( flat.coordinate[0] > 0 );
    assert( (flat.n-1) % 2 == 0 );
    size_t n = (flat.n-1)/2;
    math_core::nd_point_t p;
    p.n = n;
    p.coordinate.insert( p.coordinate.end(),
			 flat.coordinate.begin() + 1,
			 flat.coordinate.begin() + 1 + n );
    return p;
  }

  // Description:
  // encoides a netaive region into a lat point
  math_core::nd_point_t encode_negative_region( const math_core::nd_aabox_t& r )
  {
    math_core::nd_point_t res;
    res.n = 2 * r.start.n + 1;
    res.coordinate.push_back( -1.0 );
    res.coordinate.insert( res.coordinate.end(),
			   r.start.coordinate.begin(),
			   r.start.coordinate.end() );
    res.coordinate.insert( res.coordinate.end(),
			   r.end.coordinate.begin(),
			   r.end.coordinate.end() );
    return res;
  }

  // Description:
  // Decode negative region from flat point
  math_core::nd_aabox_t decode_negative_region( const math_core::nd_point_t& flat )
  {
    assert( flat.coordinate[0] > 0 );
    assert( (flat.n-1) % 2 == 0 );
    size_t n = (flat.n-1)/2;
    math_core::nd_point_t start, end;
    start.n = n;
    start.coordinate.insert( start.coordinate.end(),
			     flat.coordinate.begin() + 1,
			     flat.coordinate.begin() + 1 + n );
    end.n = n;
    end.coordinate.insert( end.coordinate.end(),
			   flat.coordinate.begin() + 1 + n,
			   flat.coordinate.end() );
    return aabox( start, end );
  }

  
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
    res.push_back( r.num_ticks );
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
    math_core::nd_vector_t vd;
    vd.n = ndim;
    vd.component.insert( vd.component.end(),
			 v.begin() + ndim,
			 v.begin() + 2 * ndim );
    r.dir = direction(vd);
    double nt = v[2*ndim];
    if( nt < 0 )
      nt = 0;
    r.num_ticks = (size_t)nt;
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
      r.num_ticks = (size_t)sample_from( uniform_distribution( 0.0, 10.0 ) );
      r.length_scale = sample_from( uniform_distribution( 0.1 * wdist,
							  0.5 * wdist) );
      r.spread = sample_from( uniform_distribution( 0.1 * r.length_scale,
						    0.5 * r.length_scale ) );
      rulers.push_back( r );
    }
    return rulers;
  }

  //====================================================================

  void gem_k_ruler_process_t::_run_GEM()
  {
    bool output = true;

    std::vector< std::vector<ruler_t> > ruler_sets;
    std::vector< std::vector<double> > mixture_sets;
    std::vector< double > liks;
    size_t best_idx = 0;
    
    // run a beunch of GEM runs
    for( size_t i = 0; i < _params.num_gem_restarts; ++i ) {

      // make sure to clear the rulers/mixtures
      // so they are initialized
      _rulers.clear();
      _ruler_mixture_weights.clear();
      _run_single_GEM();
      ruler_sets.push_back( _rulers );
      mixture_sets.push_back( _ruler_mixture_weights );
      liks.push_back( likelihood() );

      if( liks[best_idx] < liks[liks.size()-1] ) {
	best_idx = i;
      }

      if( output ) {
	std::cout << "GEM run[" << i << "] best=" << liks[best_idx] << " (" << liks[i] << ")" << std::endl;
      }
    }

    // set rulers/mixture to best likelihood from GEM runs
    _rulers = ruler_sets[best_idx];
    _ruler_mixture_weights = mixture_sets[best_idx];
  }

  //====================================================================


  void gem_k_ruler_process_t::_run_single_GEM()
  {
    if( _rulers.size() != _params.num_rulers ) {
      _rulers = create_initial_rulers();
    }
    
    // convert from rulers to double-vectors for GEM
    std::vector<std::vector<double> > ruler_params;
    for( size_t i = 0; i < _rulers.size(); ++i ) {
      ruler_params.push_back( to_flat_vector( _rulers[i] ) );
    }

    // create lower/upper bounds on parameterts for rulers
    std::vector<std::vector<double> > lb, ub;
    for( size_t i = 0; i < _rulers.size(); ++i ) {
      std::vector<double> l, u;
      l.insert( l.end(),
		_window.start.coordinate.begin(),
		_window.start.coordinate.end() );
      nd_vector_t temp = -1.0 * ( _window.end - _window.start );
      l.insert( l.end(),
		temp.component.begin(),
		temp.component.end() );
      l.push_back( 0 );
      l.push_back( 0 );
      l.push_back( 0.1 );

      double wdist = distance( _window.start, _window.end );
      u.insert( u.end(),
		_window.end.coordinate.begin(),
		_window.end.coordinate.end() );
      temp = ( _window.end - _window.start );
      u.insert( u.end(),
		temp.component.begin(),
		temp.component.end() );
      u.push_back( 30 );
      u.push_back( wdist );
      u.push_back( wdist );

      lb.push_back( l );
      ub.push_back( u );
    }

    // the resulting parameter vector
    std::vector<std::vector<double> > mle_estimate;
    std::vector<double> mle_mixtures;

    // the likeliohood function
    using std::placeholders::_1;
    using std::placeholders::_2;
    std::function<double(const math_core::nd_point_t& single_data,
			 const std::vector<double>& params)>
      lik_f = std::bind( &gem_k_ruler_process_t::lik_mixed_ruler_flat,
			 *this,
			 _1,
			 _2);

    // debug output
    std::cout << "_run_GEM: flat_rulers: " << std::endl;
    for( size_t i = 0; i < ruler_params.size(); ++i ) {
      std::cout << " [" << i << "]  ";
      for( size_t j = 0; j < ruler_params[i].size(); ++j ) {
	std::cout << ruler_params[i][j] << " , ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;

    // encode all observations and negative objservations
    // into a single vector of poiints
    std::vector<math_core::nd_point_t> data;
    for( size_t i = 0; i < _observations.size(); ++i ) {
      data.push_back( encode_point( _observations[i] ) );
    }
    // trying this HACK: since mixture models cannot
    // encode that negative observations must be taxed for every
    // model, fake out enough virtual negative observations that ther
    // is one per original per model to skew the weights
    //for( size_t j = 0; j < _rulers.size(); ++j ) {
      for( size_t i = 0; i < _negative_observations.size(); ++i ) {
	data.push_back( encode_negative_region( _negative_observations[i] ) );
      }
      //}
    
    // run GEM
    run_GEM_mixture_model_MLE_numerical
      ( _params.gem,
	data,
	ruler_params,
	lb,
	ub,
	lik_f,
	mle_estimate,
	mle_mixtures );

    // convert doubles into rulers
    for( size_t i = 0; i < _rulers.size(); ++i ) {
      _rulers[i] = to_ruler( mle_estimate[i], _ndim );
    }
    _ruler_mixture_weights = mle_mixtures;
  }

  //====================================================================

  std::vector<nd_point_t> 
  gem_k_ruler_process_t::ticks_for_ruler( const ruler_t& r ) const
  {
    std::vector<nd_point_t> ticks;
    ticks.push_back( r.start );
    for( size_t i = 0 ; i < r.num_ticks; ++i ) {
      nd_point_t t = ticks[ ticks.size() - 1] + r.length_scale * r.dir; 
      if( is_inside( t, _window ) ) {
	ticks.push_back( t );
      }
    }
    return ticks;
  }

  //====================================================================

  double gem_k_ruler_process_t::lik_single_point_single_ruler
  ( const math_core::nd_point_t& x,
    const ruler_t& ruler ) const
  {
    double bad_value = 1e-11;
    if( ruler.length_scale < ( 2 * ruler.spread * 2 ) ) {
      return bad_value;
    }

    double p = 0;
    gaussian_distribution_t gauss;
    gauss.dimension = 1;
    gauss.means = { 0.0 };
    gauss.covariance = diagonal_matrix( point( ruler.spread ) );
    std::vector<nd_point_t> ticks = ticks_for_ruler(ruler);
    // std::cout << "  lik: " << ticks.size() << "tick "
    // 	      << ruler_norm.start << ", " << ruler.dir << ", "
    // 	      << ruler_norm.length<< " (" << ruler.length << ") , " 
    // 	      << ruler_norm.length_scale << " (" << ruler.length_scale << "), "
    // 	      << ruler_norm.spread << " (" << ruler.spread << ")" << std::endl;
    for( size_t i = 0; i < ticks.size(); ++i ) {
      double dist = distance(ticks[i],x);
      double t = pdf( point(dist), gauss );
      // choose the "best" tick for this data
      if( p < t ) {
	p = t;
      }
    }
    if( std::isnan( p ) || std::isinf(p) ) {
      return bad_value;
    }
    if( p < 10 * bad_value ) {
      return 10 * bad_value;
    }
    return p;
  }

  //====================================================================

  double
  gem_k_ruler_process_t::lik_mixed_ruler_flat
  ( const math_core::nd_point_t& flat,
    const std::vector<double>& ruler_params ) const
  {
    if( flat.coordinate[0] > 0 ) {
      return lik_single_point_single_ruler_flat( decode_point(flat),
						 ruler_params);
    } else {
      return lik_negative_region_ruler_flat( decode_negative_region(flat),
					     ruler_params );
    }
  }

  //====================================================================

  static bool neg_lik_output = false;

  double 
  gem_k_ruler_process_t::lik_negative_region_ruler
  ( const math_core::nd_aabox_t& region,
    const ruler_t& ruler ) const
  {
    double bad_value = 1e-9;
    bool output = neg_lik_output;
    if( output ) {
      std::cout << "  neg-lik  reg: " << region 
		<< " ruler: " << ruler << std::endl;
    }
    //math_core::mpt::mp_float p = 0;
    math_core::mpt::mp_float p = 1;
    gaussian_distribution_t spread_distribution;
    spread_distribution.dimension = _ndim;
    spread_distribution.means = std::vector<double>( _ndim, 0.0 );
    spread_distribution.covariance = diagonal_matrix( point( std::vector<double>(_ndim,ruler.spread) ) );
    std::vector<nd_point_t> ticks = ticks_for_ruler(ruler);
    for( size_t i = 0; i < ticks.size(); ++i ) {
      nd_point_t tick_point = ticks[i];
      nd_aabox_t reg = region;
      reg.start = zero_point( _ndim ) + ( reg.start - tick_point );
      reg.end = zero_point( _ndim ) + ( reg.end - tick_point );
      
      // compute probability mass inside region and then
      // take the mass outside of region
      math_core::mpt::mp_float outside_mass = 
	( 1.0 - 
	  (cdf(reg.end,spread_distribution)
	   - cdf(reg.start,spread_distribution )));

      if( output ) {
	std::cout << "    tick[" << i << "]: " << tick_point 
		  << " reg: " << reg << " cdf=" << outside_mass
		  << " [" << cdf(reg.end,spread_distribution)
		  << " , " << cdf(reg.start,spread_distribution)
		  << "]" << std::endl;
      }

      // if( p < outside_mass ) {
      //  	p = outside_mass;
      // }
      //p += outside_mass;
      p *= outside_mass;
    }
    //p /= ticks.size(); // compute the "expected" prob over all ticks
    //p = 1.0 / p;
    if( p < 10 * bad_value ) {
      p = 10 * bad_value;
    }
    return p.convert_to<double>();
  }
  
  //====================================================================

  double gem_k_ruler_process_t::likelihood() const
  {

    bool output = false;
    //neg_lik_output = true;
    if( output ){
      std::cout << "gem_k_ruler_process::likelihood" << std::endl;
    }

    // Calculate the likelihood of the data
    math_core::mpt::mp_float p = 0;
    for( size_t i = 0; i < _observations.size(); ++i ) {
      nd_point_t x = _observations[i];
      for( size_t mix_i = 0; mix_i < _rulers.size(); ++mix_i ) {
	ruler_t ruler = _rulers[ mix_i ];
	double w = _ruler_mixture_weights[ mix_i ];
	p += w * lik_single_point_single_ruler( x, ruler );

	if( output ) {
	  std::cout << "  data[" << i << "] mix= " << w << " ruler: " << ruler << " lik" << x << "=" << lik_single_point_single_ruler(x,ruler) << std::endl;
	}

      }
    }

    // add likleihoods of negative regions
    for( size_t i = 0; i < _negative_observations.size(); ++i ) {
      nd_aabox_t reg = _negative_observations[i];
      for( size_t mix_i = 0; mix_i < _rulers.size(); ++mix_i ) {
	ruler_t ruler = _rulers[ mix_i ];
	double w = _ruler_mixture_weights[ mix_i ];
	p += w * lik_negative_region_ruler( reg, ruler );

	if( output ) {
	  std::cout << "  region[" << i << "] mix= " << w << " ruler: " << ruler << " lik" << reg << "=" << lik_negative_region_ruler(reg,ruler) << std::endl;
	}

      }
    }

    neg_lik_output = false;
    
    // return the total likelihood
    return p.convert_to<double>();
  }

  //====================================================================

  std::ostream& operator<< (std::ostream& os, const ruler_t& r )
  {
    os << "<-" << r.start << " " << r.dir 
       << " #" << r.num_ticks << " " << r.length_scale
       << " ~" << r.spread << "->";
    return os;
  }

  //====================================================================
  
  std::vector<math_core::nd_point_t> 
  gem_k_ruler_process_t::sample() const
    {
      std::vector<math_core::nd_point_t> s;
      // for each ruler, see if it is "on" by using the
      // mixture weight, and if so add the ticks
      for( size_t i = 0; i < _rulers.size(); ++i ) {
	if( probability_core::flip_coin( _ruler_mixture_weights[i] ) ) {
	  std::vector<math_core::nd_point_t> ticks
	    = ticks_for_ruler( _rulers[i] );
	  for( math_core::nd_point_t tick : ticks ) {
	    if( math_core::is_inside( tick, _window ) ) {
	      s.push_back( tick );
	    }
	  }
	}
      }
      return s;
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


}
