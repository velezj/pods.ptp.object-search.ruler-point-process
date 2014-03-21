
#include "gem_k_tick_process.hpp"
#include "flat_parameters.hpp"
#include <probability-core/distribution_utils.hpp>
#include <math-core/geom.hpp>
#include <math-core/matrix.hpp>
#include <math-core/io.hpp>
#include <math-core/mpt.hpp>
#include <stdexcept>
#include <iostream>
#include <cmath>

using namespace math_core;
using namespace probability_core;


namespace ruler_point_process {

  //====================================================================

  template<class TickModelT>
  std::vector<TickModelT>
  gem_k_tick_process_t<TickModelT>::create_initial_tick_models
  ( const double& margin ) const
  {
    // get the lower/upper bounds
    std::vector<std::vector<double> > alb 
      = _params.tick_model_flat_lower_bounds;
    std::vector<std::vector<double> > aub
      = _params.tick_model_flat_upper_bounds;
    
    std::vector<TickModelT> models;
    for( size_t i = 0; i < _params.k; ++i ) {
      std::vector<double> lb = alb[i];
      std::vector<double> ub = aub[i];
      TickModelT lower_model, upper_model;
      flat_to_model( lb, lower_model );
      flat_to_model( ub, flat_to_model );
      while( true ) {
	
	TickModelT model =
	  sample_from( uniform_distribution( lower_model, upper_model) );

	// check that we are within bounds and not too
	// much at hte edge
	std::vector<double> x = model_to_flat( model );
	if( !within_bounds_margin( x, lb, ub, margin ) ) {
	  continue; // try again
	}

	models.push_back( model );
	break;
      }
    }
    return ticks;
  }


  //====================================================================

  template< class TickModelT >
  void gem_k_tick_process_t<TickModelT>::_run_GEM()
  {
    bool output = true;

    std::vector< std::vector<TickModelT> > model_sets;
    std::vector< std::vector<double> > mixture_sets;
    std::vector< double > liks;
    size_t best_idx = 0;
    
    // run a beunch of GEM runs
    for( size_t i = 0; i < _params.num_gem_restarts; ++i ) {

      // make sure to clear the ticks/mixtures
      // so they are initialized
      _tick_models.clear();
      _mixture_weights.clear();
      _run_single_GEM();
      model_sets.push_back( _tick_models );
      mixture_sets.push_back( _mixture_weights );
      liks.push_back( likelihood() );

      if( liks[best_idx] < liks[liks.size()-1] ) {
	best_idx = i;
      }

      if( output ) {
	std::cout << "GEM run[" << i << "] best=" << liks[best_idx] << " (" << liks[i] << ")" << std::endl;
	for( size_t i = 0; i < model_sets[model_sets.size()-1].size(); ++i ) {
	  std::cout << "   model[" << i << "]: " << model_sets[model_sets.size()-1][i] << std::endl;
	}
      }
    }

    // set ticks/mixture to best likelihood from GEM runs
    _tick_models = model_sets[best_idx];
    _mixture_weights = mixture_sets[best_idx];
  }

  //====================================================================


  template<class TickModelT>
  void gem_k_tick_process_t<TickModelT>::_run_single_GEM()
  {
    if( _tick_models.size() != _params.k ) {
      _tick_models = create_initial_tick_models();
    }
    
    // convert from ticks to double-vectors for GEM
    std::vector<std::vector<double> > model_flat_params;
    for( size_t i = 0; i < _tick_models.size(); ++i ) {
      model_flat_params.push_back( model_to_flat( _tick_models[i] ) );
    }

    // the resulting parameter vector
    std::vector<std::vector<double> > mle_estimate;
    std::vector<double> mle_mixtures;

    // the likeliohood function
    using std::placeholders::_1;
    using std::placeholders::_2;
    std::function<double(const math_core::nd_point_t& single_data,
			 const std::vector<double>& params)>
      lik_f = std::bind( &gem_k_tick_process_t<TickModelT>::lik_mixed_flat,
			 *this,
			 _1,
			 _2);

    // debug output
    std::cout << "_run_GEM: flat_models: " << std::endl;
    for( size_t i = 0; i < flat_model_params.size(); ++i ) {
      std::cout << " [" << i << "]  ";
      for( size_t j = 0; j < flat_model_params[i].size(); ++j ) {
	std::cout << flat_model_params[i][j] << " , ";
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
	flat_model_params,
	_params.tick_model_flat_lower_bounds,
	_params.tick_model_flat_upper_bounds,
	lik_f,
	mle_estimate,
	mle_mixtures );

    // convert doubles into ticks
    for( size_t i = 0; i < _ticks.size(); ++i ) {
      TickModelT model;
      flat_to_model( mle_estimate[i], model, _params.tick_model_parameters );
      _tick_models[i] = model;
    }
    _mixture_weights = mle_mixtures;
  }

  //====================================================================

  template<class TickModelT>
  double gem_k_tick_process_t<TickModelT>::lik_single_point_single_model
  ( const math_core::nd_point_t& x,
    const TickModelT& model ) const
  {

    double p = 0;
    gaussian_distribution_t gauss;
    gauss.dimension = 1;
    gauss.means = { 0.0 };
    gauss.covariance = diagonal_matrix( point( tick.spread ) );
    std::vector<nd_point_t> ticks = ticks_for_model(model);
    for( size_t i = 0; i < ticks.size(); ++i ) {
      double dist = distance(ticks[i],x);
      double t = pdf( point(dist), gauss );
      p += t;
    }
    if( std::isnan( p ) || std::isinf(p) ) {
      return bad_value;
    }
    if( p < _params.min_likelihood_epsilon ) {
      p = _params.min_likelihood_epsilon;
    }
    return p;
  }

  //====================================================================

  template<class TickModelT>
  double
  gem_k_tick_process_t<TickModelT>::lik_mixed_flat
  ( const math_core::nd_point_t& flat,
    const std::vector<double>& model_params ) const
  {
    if( flat.coordinate[0] > 0 ) {
      return lik_single_point_single_model_flat( decode_point(flat),
						 model_params);
    } else {
      return lik_negative_region_flat( decode_negative_region(flat),
				       model_params );
    }
  }

  //====================================================================

  static bool neg_lik_output = false;

  template<class TickModelT>
  double 
  gem_k_tick_process_t<TickModelT>::lik_negative_region_tick
  ( const math_core::nd_aabox_t& region,
    const TickModelT& model ) const
  {
    bool output = neg_lik_output;
    double p = 1;
    gaussian_distribution_t spread_distribution;
    spread_distribution.dimension = _params.point_dimension;
    spread_distribution.means 
      = std::vector<double>( _params.point_dimension, 0.0 );
    spread_distribution.covariance 
      = diagonal_matrix( point( std::vector<double>(_params.point_dimension,
						    _params.tick_spread) ) );
    std::vector<nd_point_t> ticks = ticks_for_model(model);
    for( size_t i = 0; i < ticks.size(); ++i ) {
      nd_point_t tick_point = ticks[i];
      nd_aabox_t reg = region;
      reg.start = zero_point( _ndim ) + ( reg.start - tick_point );
      reg.end = zero_point( _ndim ) + ( reg.end - tick_point );

      double c0 = cdf(reg.end,spread_distribution);
      double c1 = cdf(reg.start,spread_distribution );

      if( c0 < c1 ) {
	double temp = c1;
	c1 = c0;
	c0 = temp;
      }
      
      // compute probability mass inside region and then
      // take the mass outside of region
      math_core::mpt::mp_float outside_mass = 
	( 1.0 - 
	  (c0 - c1));

      p *= outside_mass;
    }

    // return p (limited by epsilon)
    double bad_value = _params.min_likelihood_epsilon;
    if( p < bad_value ) {
      p = bad_value;
    }
    if( std::isnan(p) || std::isinf(p) ) {
      return bad_value;
    }
    return p;
  }
  
  //====================================================================

  template<class TickModelT>
  double gem_k_tick_process_t<TickModelT>::likelihood() const
  {

    bool output = false;
    //neg_lik_output = true;
    if( output ){
      std::cout << "gem_k_tick_process::likelihood" << std::endl;
    }

    // Calculate the likelihood of the data
    math_core::mpt::mp_float p = 0;
    for( size_t i = 0; i < _observations.size(); ++i ) {
      nd_point_t x = _observations[i];
      for( size_t mix_i = 0; mix_i < _ticks.size(); ++mix_i ) {
	TickModelT model = _tick_models[ mix_i ];
	double w = _mixture_weights[ mix_i ];
	p += w * lik_single_point_single_model( x, model );

	if( output ) {
	  std::cout << "  data[" << i << "] mix= " << w << " model: " << model << " lik" << x << "=" << lik_single_point_single_model(x,tick) << std::endl;
	}

      }
    }

    // add likleihoods of negative regions
    for( size_t i = 0; i < _negative_observations.size(); ++i ) {
      nd_aabox_t reg = _negative_observations[i];
      for( size_t mix_i = 0; mix_i < _ticks.size(); ++mix_i ) {
	TickModelT model = _tick_models[ mix_i ];
	double w = _mixture_weights[ mix_i ];
	p += w * lik_negative_region( reg, model );

	if( output ) {
	  std::cout << "  region[" << i << "] mix= " << w << " model: " << model << " lik" << reg << "=" << lik_negative_region(reg,model) << std::endl;
	}

      }
    }

    neg_lik_output = false;
    
    // return the total likelihood
    return p.convert_to<double>();
  }

  //====================================================================

  template<class TickModelT>
  std::vector<math_core::nd_point_t> 
  gem_k_tick_process_t<TickModelT>::sample() const
    {
      std::vector<math_core::nd_point_t> s;
      // for each tick, see if it is "on" by using the
      // mixture weight, and if so add the ticks
      for( size_t i = 0; i < _ticks.size(); ++i ) {
	if( probability_core::flip_coin( _mixture_weights[i] ) ) {
	  std::vector<math_core::nd_point_t> ticks
	    = ticks_for_tick( _tick_models[i] );
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
