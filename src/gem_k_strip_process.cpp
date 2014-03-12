
#include "gem_k_strip_process.hpp"
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


// UTILITY functions out of the namespace

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
    assert( flat.coordinate[0] < 0 );
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



namespace ruler_point_process {


  
  //====================================================================

  std::vector<double> 
  gem_k_strip_process_t::to_flat_vector( const strip_t& r )
  {
    std::vector<double> res;
    res.insert(res.end(), 
	       r.start.coordinate.begin(),
	       r.start.coordinate.end());
    std::vector<double> c = r.manifold.coefficients();
    res.insert( res.end(),
		c.begin(),
		c.end() );
    res.push_back( r.num_ticks );
    res.push_back( r.length_scale );
    res.push_back( r.spread );
    return res;
  }

  //====================================================================
  
  strip_t 
  gem_k_strip_process_t::to_strip( const std::vector<double>& v,
				   const size_t ndim,
				   const size_t poly_dim)
  {
    assert( v.size() >= (ndim + poly_dim + 1 + 2 + 1) );
    if( v.size() < ( ndim + poly_dim + 1 + 2 + 1) ) {
      //BOOST_THROW_EXCEPTION( flatten_error() );
      throw std::length_error("cannot unpack flat vector to strip: to few elements!");
    }
    strip_t r;
    r.start = point( ndim, v.data() );
    std::vector<double> coeffs( v.begin() + ndim,
				v.begin() + ndim + poly_dim + 1 );
    r.manifold = polynomial_t(coeffs);
    double nt = v[ndim + poly_dim + 1];
    if( nt < 0 )
      nt = 0;
    r.num_ticks = (size_t)nt;
    r.length_scale = v[ndim+poly_dim+1 + 1];
    r.spread = v[ndim+poly_dim+1 + 2];
    return r;
  }


  //====================================================================

  std::vector<strip_t>
  gem_k_strip_process_t::create_initial_strips() const
  {
    // create the lower/upper bounds
    std::vector<std::vector<double> > alb;
    std::vector<std::vector<double> > aub;
    create_bounds( _params.num_strips, alb, aub );

    std::vector<strip_t> strips;
    for( size_t i = 0; i < _params.num_strips; ++i ) {
      std::vector<double> lb = alb[i];
      std::vector<double> ub = aub[i];
      while( true ) {
	strip_t r;
	r.start = sample_from( uniform_distribution( _window.start,
						     _window.end ) );
	std::vector<double> coeffs;
	for( size_t i = 0; i < _params.strip_poly_order + 1; ++i ) {
	  coeffs.push_back( sample_from( uniform_distribution( -90.0, 90.0 ) ) );
	}
	r.manifold = math_core::polynomial_t( coeffs );
	double wdist = distance( _window.start,
				 _window.end );
	r.num_ticks = (size_t)sample_from( uniform_distribution( 1.0, 10.0 ) );
	r.length_scale = sample_from( uniform_distribution( 0.1 * wdist,
							    0.5 * wdist) );
	r.spread = sample_from( uniform_distribution( 0.1,
						      0.5 ) );

	// check that we are within bounds and not too
	// much at hte edge
	std::vector<double> x = to_flat_vector( r );
	for( size_t i = 0; i < x.size(); ++i ) {
	  double br = ub[i] - lb[i];
	  if( x[i] <= lb[i] + 0.1 * br ||
	      x[i] >= ub[i] - 0.1 * br ) {
	    continue; // try again
	  }
	}

	strips.push_back( r );
	break;
      }
    }
    return strips;
  }

  //====================================================================

  void gem_k_strip_process_t::create_bounds
  ( const size_t& num_strips,
    std::vector<std::vector<double> >& lb,
    std::vector<std::vector<double> >& ub ) const
  {
    // create lower/upper bounds on parameterts for strips
    for( size_t i = 0; i < num_strips; ++i ) {
      std::vector<double> l, u;
      l.insert( l.end(),
		_window.start.coordinate.begin(),
		_window.start.coordinate.end() );
      for( size_t i = 0; i < _params.strip_poly_order + 1; ++i ) {
	l.push_back( -100.0 );
      }
      l.push_back( 0 );
      l.push_back( 0 );
      l.push_back( 0.1 );

      double wdist = distance( _window.start, _window.end );
      u.insert( u.end(),
		_window.end.coordinate.begin(),
		_window.end.coordinate.end() );
      for( size_t i = 0; i < _params.strip_poly_order + 1; ++i ) {
	u.push_back( 100.0 );
      }
      u.push_back( 30 );
      u.push_back( wdist );
      u.push_back( wdist );

      lb.push_back( l );
      ub.push_back( u );
    }    
  }

  //====================================================================


  void gem_k_strip_process_t::_run_GEM()
  {
    bool output = true;

    std::vector< std::vector<strip_t> > strip_sets;
    std::vector< std::vector<double> > mixture_sets;
    std::vector< double > liks;
    size_t best_idx = 0;
    
    // run a beunch of GEM runs
    for( size_t i = 0; i < _params.num_gem_restarts; ++i ) {

      // make sure to clear the strips/mixtures
      // so they are initialized
      _strips.clear();
      _mixture_weights.clear();
      _run_single_GEM();
      strip_sets.push_back( _strips );
      mixture_sets.push_back( _mixture_weights );
      liks.push_back( likelihood() );

      if( liks[best_idx] < liks[liks.size()-1] ) {
	best_idx = i;
      }

      if( output ) {
	std::cout << "GEM run[" << i << "] best=" << liks[best_idx] << " (" << liks[i] << ")" << std::endl;
	for( size_t i = 0; i < strip_sets[strip_sets.size()-1].size(); ++i ) {
	  std::cout << "   strip[" << i << "]: " << strip_sets[strip_sets.size()-1][i] << std::endl;
	}
      }
    }

    // set strips/mixture to best likelihood from GEM runs
    _strips = strip_sets[best_idx];
    _mixture_weights = mixture_sets[best_idx];
  }

  //====================================================================


  void gem_k_strip_process_t::_run_single_GEM()
  {
    if( _strips.size() != _params.num_strips ) {
      _strips = create_initial_strips();
    }
    
    // convert from strips to double-vectors for GEM
    std::vector<std::vector<double> > strip_params;
    for( size_t i = 0; i < _strips.size(); ++i ) {
      strip_params.push_back( to_flat_vector( _strips[i] ) );
    }

    // get the bounds
    std::vector<std::vector<double> > lb, ub;
    create_bounds( _params.num_strips, lb, ub );

    // the resulting parameter vector
    std::vector<std::vector<double> > mle_estimate;
    std::vector<double> mle_mixtures;

    // the likeliohood function
    using std::placeholders::_1;
    using std::placeholders::_2;
    std::function<double(const math_core::nd_point_t& single_data,
			 const std::vector<double>& params)>
      lik_f = std::bind( &gem_k_strip_process_t::lik_mixed_strip_flat,
			 *this,
			 _1,
			 _2);

    // debug output
    std::cout << "_run_GEM: flat_strips: " << std::endl;
    for( size_t i = 0; i < strip_params.size(); ++i ) {
      std::cout << " [" << i << "]  ";
      for( size_t j = 0; j < strip_params[i].size(); ++j ) {
	std::cout << strip_params[i][j] << " , ";
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
    for( size_t i = 0; i < _negative_observations.size(); ++i ) {
      data.push_back( encode_negative_region( _negative_observations[i] ) );
    }
       
    // run GEM
    run_GEM_mixture_model_MLE_numerical
      ( _params.gem,
	data,
	strip_params,
	lb,
	ub,
	lik_f,
	mle_estimate,
	mle_mixtures );

    // convert doubles into strips
    for( size_t i = 0; i < _strips.size(); ++i ) {
      _strips[i] = to_strip( mle_estimate[i], _ndim, _params.strip_poly_order);
    }
    _mixture_weights = mle_mixtures;
  }

  //====================================================================

  std::vector<nd_point_t> 
  gem_k_strip_process_t::ticks_for_strip( const strip_t& r ) const
  {
    assert( _ndim == 2 );
    if( _ndim != 2 ) {
      BOOST_THROW_EXCEPTION( std::logic_error( "cannot handle polynomials of dimension other than 2") );
    }
    std::vector<nd_point_t> ticks;
    ticks.push_back( r.start );
    double start_x = r.start.coordinate[0];
    for( size_t i = 0 ; i < r.num_ticks; ++i ) {
      double tx = r.manifold.find_point_arc_length_away_chord_approx( start_x,
								      r.length_scale * (i+1) );
      nd_point_t tp = point( tx,
			     r.manifold.evaluate(tx));
      if( is_inside( tp, _window ) ) {
	ticks.push_back( tp );
      }
    }
    return ticks;
  }

  //====================================================================

  double gem_k_strip_process_t::lik_single_point_single_strip
  ( const math_core::nd_point_t& x,
    const strip_t& strip ) const
  {
    double bad_value = 1e-11;
    if( strip.length_scale < ( 2 * strip.spread * 2 ) ) {
      return bad_value;
    }

    double p = 0;
    gaussian_distribution_t gauss;
    gauss.dimension = 1;
    gauss.means = { 0.0 };
    gauss.covariance = diagonal_matrix( point( strip.spread ) );
    std::vector<nd_point_t> ticks = ticks_for_strip(strip);
    // std::cout << "  lik: " << ticks.size() << "tick "
    // 	      << strip.start << ", " << strip.manifold << ", "
    // 	      << strip.num_ticks << ", "
    // 	      << strip.length_scale << ", "
    // 	      << strip.spread << std::endl;
    for( size_t i = 0; i < ticks.size(); ++i ) {
      double dist = distance(ticks[i],x);
      double t = pdf( point(dist), gauss );
      // // choose the "best" tick for this data
      // if( p < t ) {
      // 	p = t;
      // }
      p += t;
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
  gem_k_strip_process_t::lik_mixed_strip_flat
  ( const math_core::nd_point_t& flat,
    const std::vector<double>& strip_params ) const
  {
    if( flat.coordinate[0] > 0 ) {
      return lik_single_point_single_strip_flat( decode_point(flat),
						 strip_params);
    } else {
      return lik_negative_region_strip_flat( decode_negative_region(flat),
					     strip_params );
    }
  }

  //====================================================================

  static bool neg_lik_output = false;

  double 
  gem_k_strip_process_t::lik_negative_region_strip
  ( const math_core::nd_aabox_t& region,
    const strip_t& strip ) const
  {
    double bad_value = 1e-19;
    bool output = neg_lik_output;
    if( output ) {
      std::cout << "  neg-lik  reg: " << region 
		<< " strip: " << strip << std::endl;
    }
    //math_core::mpt::mp_float p = 0;
    math_core::mpt::mp_float p = 1;
    gaussian_distribution_t spread_distribution;
    spread_distribution.dimension = _ndim;
    spread_distribution.means = std::vector<double>( _ndim, 0.0 );
    spread_distribution.covariance = diagonal_matrix( point( std::vector<double>(_ndim,strip.spread) ) );
    std::vector<nd_point_t> ticks = ticks_for_strip(strip);
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
    if( p < bad_value ) {
      p = bad_value;
    }
    double ret = p.convert_to<double>();
    if( std::isnan(ret) || std::isinf(ret)  ) {
      return bad_value;
    }
    return ret;
  }
  
  //====================================================================

  double gem_k_strip_process_t::likelihood() const
  {

    bool output = false;
    //neg_lik_output = true;
    if( output ){
      std::cout << "gem_k_strip_process::likelihood" << std::endl;
    }

    // Calculate the likelihood of the data
    math_core::mpt::mp_float p = 0;
    for( size_t i = 0; i < _observations.size(); ++i ) {
      nd_point_t x = _observations[i];
      for( size_t mix_i = 0; mix_i < _strips.size(); ++mix_i ) {
	strip_t strip = _strips[ mix_i ];
	double w = _mixture_weights[ mix_i ];
	p += w * lik_single_point_single_strip( x, strip );

	if( output ) {
	  std::cout << "  data[" << i << "] mix= " << w << " strip: " << strip << " lik" << x << "=" << lik_single_point_single_strip(x,strip) << std::endl;
	}

      }
    }

    // add likleihoods of negative regions
    for( size_t i = 0; i < _negative_observations.size(); ++i ) {
      nd_aabox_t reg = _negative_observations[i];
      for( size_t mix_i = 0; mix_i < _strips.size(); ++mix_i ) {
	strip_t strip = _strips[ mix_i ];
	double w = _mixture_weights[ mix_i ];
	p += w * lik_negative_region_strip( reg, strip );

	if( output ) {
	  std::cout << "  region[" << i << "] mix= " << w << " strip: " << strip << " lik" << reg << "=" << lik_negative_region_strip(reg,strip) << std::endl;
	}

      }
    }

    neg_lik_output = false;
    
    // return the total likelihood
    return p.convert_to<double>();
  }

  //====================================================================

  std::ostream& operator<< (std::ostream& os, const strip_t& r )
  {
    os << "<~" << r.start << " " << r.manifold 
       << " #" << r.num_ticks << " " << r.length_scale
       << " ~ " << r.spread << "~>";
    return os;
  }

  //====================================================================
  
  std::vector<math_core::nd_point_t> 
  gem_k_strip_process_t::sample() const
    {
      std::vector<math_core::nd_point_t> s;
      // for each strip, see if it is "on" by using the
      // mixture weight, and if so add the ticks
      for( size_t i = 0; i < _strips.size(); ++i ) {
	if( probability_core::flip_coin( _mixture_weights[i] ) ) {
	  std::vector<math_core::nd_point_t> ticks
	    = ticks_for_strip( _strips[i] );
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
