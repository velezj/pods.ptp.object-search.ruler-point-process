
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

  static bool within_bounds_margin( const std::vector<double>& x,
				    const std::vector<double>& lower,
				    const std::vector<double>& upper,
				    const double& margin )
  {
    assert( x.size() == lower.size() );
    assert( x.size() == upper.size() );
    for( size_t i = 0; i < x.size(); ++i ) {
      if( x[i] - lower[i] < margin ) {
	return false;
      }
      if( upper[i] - x[i] < margin ) {
	return false;
      }
    }
    return true;
  }

  //====================================================================

  template<class TickModelT>
  std::vector<TickModelT>
  gem_k_tick_process_t<TickModelT>::create_initial_tick_models
  ( const double& margin ) const
  {
    // debug
    //std::cout << "create_initial_tick_models: started" << std::endl;
    
    // get the lower/upper bounds
    std::vector<std::vector<double> > alb 
      = _params.tick_model_flat_lower_bounds;
    std::vector<std::vector<double> > aub
      = _params.tick_model_flat_upper_bounds;
    
    std::vector<TickModelT> models;
    for( size_t i = 0; i < _params.k; ++i ) {

      // debug
      //std::cout << "starting init model " << i <<  std::endl;
      
      std::vector<double> lb = alb[i];
      std::vector<double> ub = aub[i];
      TickModelT lower_model, upper_model;
      flat_to_model( lb, lower_model, _params.tick_model_parameters );
      flat_to_model( ub, upper_model, _params.tick_model_parameters );
      while( true ) {
	
	TickModelT model =
	  sample_from( uniform_distribution( lower_model, upper_model) );

	// check that we are within bounds and not too
	// much at hte edge
	std::vector<double> x = model_to_flat( model, _params.tick_model_parameters );

	// debug
	// std::cout << "   sampling initial model " << i << ": ";
	// for( double d : x ){
	//   std::cout << d << ", ";
	// }
	// std::cout << std::endl;
	
	if( !within_bounds_margin( x, lb, ub, margin ) ) {

	  // debug
	  std::cout << "  outside margin=" << margin << std::endl;
	  std::cout << "  LB: ";
	  for( size_t i = 0; i < x.size(); ++i ) {
	    std::cout << x[i]-lb[i] << ", ";
	  }
	  std::cout << std::endl;
	  std::cout << "  UB: ";
	  for( size_t i = 0; i < x.size(); ++i ) {
	    std::cout << ub[i]-x[i] << ", ";
	  }
	  std::cout << std::endl;
	  continue; // try again
	}

	models.push_back( model );
	break;
      }
    }
    return models;
  }


  //====================================================================

  template< class TickModelT >
  void gem_k_tick_process_t<TickModelT>::_run_GEM()
  {
    bool output = true;

    // debug
    //std::cout << "_run_GEM: started" << std::endl;

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
	std::cout << "      w[" << i << "]: ";
	for( double w : _mixture_weights ) {
	  std::cout << w << " , ";
	}
	std::cout << std::endl;
	for( size_t i = 0; i < model_sets[model_sets.size()-1].size(); ++i ) {
	  std::cout << "   model[" << i << "]: " << model_sets[model_sets.size()-1][i] << std::endl;
	  std::cout << "   ticks[" << i << "]: ";
	  std::vector<nd_point_t> ticks = ticks_for_model( model_sets[model_sets.size()-1][i], _params.tick_model_parameters );
	  for( nd_point_t tick : ticks ) {
	    std::cout << tick << " , ";
	  }
	  std::cout << std::endl;
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
    // debug
    //std::cout << "_run_single_GEM: started" << std::endl;
    
    if( _tick_models.size() != _params.k ) {
      _tick_models = create_initial_tick_models();
    }
    
    // convert from ticks to double-vectors for GEM
    std::vector<std::vector<double> > model_flat_params;
    for( size_t i = 0; i < _tick_models.size(); ++i ) {
      model_flat_params.push_back( model_to_flat( _tick_models[i], _params.tick_model_parameters ) );
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
    for( size_t i = 0; i < model_flat_params.size(); ++i ) {
      std::cout << " [" << i << "]  ";
      for( size_t j = 0; j < model_flat_params[i].size(); ++j ) {
	std::cout << model_flat_params[i][j] << " , ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;

    // encode all observations and negative objservations
    // into a single vector of poiints
    std::vector<math_core::nd_point_t> data;
    for( size_t i = 0; i < _observations.size(); ++i ) {
      data.push_back( encode_point( _observations[i] ) );
      //std::cout << " encoded obs: " << data[data.size()-1] << std::endl;
    }
    for( size_t i = 0; i < _negative_observations.size(); ++i ) {
      data.push_back( encode_negative_region( _negative_observations[i] ) );
      //std::cout << " encoded region: " << data[data.size()-1] << std::endl;
    }
       
    // run GEM
    run_GEM_mixture_model_MLE_numerical
      ( _params.gem,
	data,
	model_flat_params,
	_params.tick_model_flat_lower_bounds,
	_params.tick_model_flat_upper_bounds,
	lik_f,
	mle_estimate,
	mle_mixtures );

    // convert doubles into ticks
    for( size_t i = 0; i < _tick_models.size(); ++i ) {
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

    double bad_value = 1e-10;
    double p = 0;
    gaussian_distribution_t gauss;
    gauss.dimension = 1;
    gauss.means = { 0.0 };
    gauss.covariance = diagonal_matrix( point( _params.tick_spread ) );
    std::vector<nd_point_t> ticks = ticks_for_model(model,_params.tick_model_parameters);
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
  gem_k_tick_process_t<TickModelT>::lik_negative_region
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
    std::vector<nd_point_t> ticks = ticks_for_model(model,_params.tick_model_parameters);

    // debug
    // std::cout << "lik_negative_region: dim=" << spread_distribution.dimension << std::endl;
    // std::cout << "lik_negative_region: spread=" << spread_distribution << std::endl;
    // std::cout << "  ticks (#=" << ticks.size() << ") : ";
    // size_t max_output_ticks = 10;
    // for( size_t i = 0; i < std::max( ticks.size(), max_output_ticks ); ++i ) {
    //   std::cout << p << " , ";
    // }
    // std::cout << std::endl;
    
    for( size_t i = 0; i < ticks.size(); ++i ) {
      nd_point_t tick_point = ticks[i];
      nd_aabox_t reg = region;
      reg.start = zero_point( _params.point_dimension ) + ( reg.start - tick_point );
      reg.end = zero_point( _params.point_dimension ) + ( reg.end - tick_point );

      double c0 = cdf(reg.end,spread_distribution);
      double c1 = cdf(reg.start,spread_distribution );

      if( c0 < c1 ) {
	double temp = c1;
	c1 = c0;
	c0 = temp;
      }
      
      // compute probability mass inside region and then
      // take the mass outside of region
      double outside_mass = 
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
  double
  gem_k_tick_process_t<TickModelT>::lik_single_point_single_model_flat
  ( const math_core::nd_point_t& single_x,
    const std::vector<double>& params ) const
  {
    TickModelT model;
    flat_to_model( params, model, _params.tick_model_parameters );
    return lik_single_point_single_model( single_x, model );
  }


  //====================================================================

  template<class TickModelT>
  double
  gem_k_tick_process_t<TickModelT>::lik_negative_region_flat
  ( const math_core::nd_aabox_t& region,
    const std::vector<double>& params ) const
  {
    TickModelT model;
    flat_to_model( params, model, _params.tick_model_parameters);
    return lik_negative_region( region, model );
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
      for( size_t mix_i = 0; mix_i < _tick_models.size(); ++mix_i ) {
	TickModelT model = _tick_models[ mix_i ];
	double w = _mixture_weights[ mix_i ];
	p += w * lik_single_point_single_model( x, model );

	if( output ) {
	  std::cout << "  data[" << i << "] mix= " << w << " model: " << model << " lik" << x << "=" << lik_single_point_single_model(x,model) << std::endl;
	}

      }
    }

    // add likleihoods of negative regions
    for( size_t i = 0; i < _negative_observations.size(); ++i ) {
      nd_aabox_t reg = _negative_observations[i];
      for( size_t mix_i = 0; mix_i < _tick_models.size(); ++mix_i ) {
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
      // for each model, see if it is "on" by using the
      // mixture weight, and if so add the ticks
      for( size_t i = 0; i < _tick_models.size(); ++i ) {
	if( probability_core::flip_coin( _mixture_weights[i] ) ) {
	  std::vector<math_core::nd_point_t> ticks
	    = ticks_for_model( _tick_models[i],_params.tick_model_parameters );
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
