
#include "model.hpp"
#include "mcmc.hpp"
#include <math-core/io.hpp>
#include <math-core/matrix.hpp>
#include <probability-core/distribution_utils.hpp>
#include <point-process-core/context.hpp>
#include <iostream>
#include <sstream>
#include <boost/filesystem.hpp>
#include <iomanip>


#define PRINT_SAMPLE_DEBUG false

using namespace math_core;
using namespace probability_core;
using namespace point_process_core;

namespace ruler_point_process {


  //======================================================================

  std::vector<nd_point_t> 
  points_for_mixture( const ruler_point_process_state_t& state,
  		      const size_t mixture_i )
  {
    
    std::vector<nd_point_t> points;
    std::vector<size_t> indices = state.mixture_to_observation_indices[ mixture_i ];
    for( size_t i = 0; i < indices.size(); ++i ) {
      points.push_back( state.observations[ indices[i] ] );
    }
    
    return points;
  }


  //======================================================================

  std::ostream& operator<< (std::ostream& os,
  			    const ruler_point_process_state_t& state )
  {
    os << "RULER STATE: " << std::endl;
    os << state.model << std::endl;
    os << "  observations: " << std::endl;
    for( size_t i = 0; i < state.observations.size(); ++i ) {
      os << "    " << state.observations[i] << " -> " << state.observation_to_mixture[i] << std::endl;
    }
    os << std::endl;

    os << "  negative regions: " << std::endl;
    for( size_t i = 0; i < state.negative_observations.size(); ++i ) {
      os << "    " << state.negative_observations[i] << std::endl;
    }
    os << std::endl;
    
    for( size_t mixture_i = 0; mixture_i < state.mixture_gaussians.size(); ++mixture_i ) {
      os << "  Mixture " << mixture_i << ": " << std::endl;
      std::vector<nd_point_t> points = points_for_mixture( state, mixture_i );
      for( size_t i = 0; i < points.size(); ++i ) {
  	os << "      point[" << i << "]: " << points[i] << std::endl;
      }
      os << "    spread: " << state.mixture_gaussians[ mixture_i ] << std::endl;
      os << "    period: " << state.mixture_period_gammas[ mixture_i ] << std::endl;
      os << "    start:  " << state.mixture_ruler_start_gaussians[ mixture_i ] << std::endl;
      os << "    length: " << state.mixture_ruler_length_gammas[ mixture_i ] << std::endl;
      os << "    direct: " << state.mixture_ruler_direction_gaussians[ mixture_i ] << std::endl;
    }
    
    return os;
  }

  //======================================================================

  std::ostream& operator<< (std::ostream& os,
  			    const ruler_point_process_model_t& model )
  {
    os << "model alpha: " << model.alpha << std::endl;
    os << "model tick precision: " << model.precision_distribution << std::endl;
    os << "model period: " << model.period_distribution << std::endl;
    os << "model length: " << model.ruler_length_distribution << std::endl;
    os << "model start mean: " << model.ruler_start_mean_distribution << std::endl;
    os << "model start precision: " << model.ruler_start_precision_distribution << std::endl;
    os << "model dir mean: " << model.ruler_direction_mean_distribution << std::endl;
    os << "model dir precision: " << model.ruler_direction_precision_distribution << std::endl;
    return os;
  }

  //======================================================================

  void trace_samples( const ruler_point_process_state_t& state,
		      const std::vector<nd_point_t>& points )
  {
    std::ostringstream oss;
    oss << state.trace_samples_dir << "/" << context_filename( "samples.trace" );
    std::string filename = oss.str();
    boost::filesystem::create_directories( boost::filesystem::path( filename ).parent_path() );
    std::ofstream fout_trace( filename.c_str(), std::ios_base::app | std::ios_base::out );
    fout_trace << state.iteration << " " 
	       << points.size() << " ";
    for( size_t i = 0; i < points.size(); ++i ) {
      fout_trace << points[i] << " ";
    }
    fout_trace << std::endl;
    fout_trace.flush();
  }

  //======================================================================

  std::vector<nd_point_t> sample_from_correct( const ruler_point_process_state_t& state )
  {

    scoped_context_switch context( chain_context( context_t( "ruler-point-process-sample" ) ) ) ;
    
    std::vector<nd_point_t> points;
    
    // sample points from each mixture separately
    for( size_t mixture_i = 0; mixture_i < state.mixture_gaussians.size(); ++mixture_i ) {

      gaussian_distribution_t spread_distribution = state.mixture_gaussians[ mixture_i ];
      gaussian_distribution_t start_distribution = state.mixture_ruler_start_gaussians[ mixture_i ];
      gamma_distribution_t period_distribution = state.mixture_period_gammas[ mixture_i ];
      gamma_distribution_t ruler_length_distribution = state.mixture_ruler_length_gammas[ mixture_i ];
      gaussian_distribution_t ruler_direction_distribution = state.mixture_ruler_direction_gaussians[ mixture_i ];
      
      // ok, sample the starting ruler lcoaiton and direction
      nd_point_t ruler_start = sample_from( start_distribution );
      nd_direction_t ruler_direction = direction( sample_from(ruler_direction_distribution) - zero_point( ruler_start.n ) );
      
      // sample a period and a ruler length
      double period = sample_from( period_distribution );
      double ruler_length = sample_from( ruler_length_distribution );
      int num_ticks = (int)floor( ruler_length / period );

      // debug
      if( PRINT_SAMPLE_DEBUG ) {
	std::cout << std::setw(26) << " ";
	std::cout << "{" << mixture_i << "} " << ruler_start << " " << period << " " << ruler_length << " " << ruler_direction << " #ticks: " << num_ticks << std::endl;
      }

      // Ok, now for each point starting at the start of ruler
      // and going for every period, sample a point using hte spread
      for( int tick = 0; tick < num_ticks; ++tick ) {
	
  	// calculate hte tick location
  	nd_point_t tick_point = ruler_start + tick * period * ruler_direction;
	
  	// now sample a variacen about the point
  	nd_point_t p =  sample_from( spread_distribution ) + (tick_point - zero_point(tick_point.n));

	// debug
	if( PRINT_SAMPLE_DEBUG ) {
	  std::cout << std::setw(26) << " " << p << " [" << floor( p.coordinate[0] ) << "]" << std::endl;
	}
	
  	// add to point set
	if( is_inside( p, state.window ) ) {
	  points.push_back( p );
	}
      }
    }


    // sample from a whole new mixture with some probability
    double new_mixture_probability =
      state.model.alpha / 
      ( state.model.alpha + state.mixture_gaussians.size() - 1 );
    if( flip_coin( new_mixture_probability ) ) {
      
      // sample new mixture component from priors
      int dim = state.model.ruler_start_mean_distribution.dimension;
      gaussian_distribution_t zero_gaussian;
      zero_gaussian.dimension = dim;
      zero_gaussian.means = zero_point( dim ).coordinate;
      zero_gaussian.covariance = to_dense_mat( Eigen::MatrixXd::Identity( dim,dim ) * 1e-10 );
      gaussian_distribution_t spread_distribution = 
  	sample_gaussian_from( zero_gaussian,
  			      state.model.precision_distribution );
      gaussian_distribution_t start_distribution = 
  	sample_gaussian_from( state.model.ruler_start_mean_distribution,
  			      state.model.ruler_start_precision_distribution );
      gamma_distribution_t period_distribution = 
  	sample_gamma_from( state.model.period_distribution );
      gamma_distribution_t ruler_length_distribution = 
  	sample_gamma_from( state.model.ruler_length_distribution );
      gaussian_distribution_t ruler_direction_distribution = 
  	sample_gaussian_from( state.model.ruler_direction_mean_distribution,
  			      state.model.ruler_direction_precision_distribution );

      // ok, sample the starting ruler lcoaiton and direction
      nd_point_t ruler_start = sample_from( start_distribution );
      nd_direction_t ruler_direction = direction( sample_from(ruler_direction_distribution) - zero_point( ruler_start.n ) );
      
      // sample a period and a ruler length
      double period = sample_from( period_distribution );
      double ruler_length = sample_from( ruler_length_distribution );
      int num_ticks = (int)floor( ruler_length / period );

      // debug
      if( PRINT_SAMPLE_DEBUG ) {
	std::cout << std::setw(26) << " ";
	std::cout << "{NEW} " << ruler_start << " " << period << " " << ruler_length << " " << ruler_direction << " #ticks: " << num_ticks << std::endl;
      }


      // Ok, now for each point starting at the start of ruler
      // and going for every period, sample a point using hte spread
      for( int tick = 0; tick < num_ticks; ++tick ) {
	
  	// calculate hte tick location
  	nd_point_t tick_point = ruler_start + tick * period * ruler_direction;
	
  	// now sample a variacen about the point
  	nd_point_t p = sample_from( spread_distribution ) + (tick_point - zero_point(dim));

	// debug
	if( PRINT_SAMPLE_DEBUG ) {
	  std::cout << std::setw(26) << " " << p << " [" << floor( p.coordinate[0] ) << "]" << std::endl;
	}
	
  	// add to point set
	if( is_inside( p, state.window ) ) {
	  points.push_back( p );
	}
      }
      
    }

    // Ok, now add all of the points already known
    points.insert( points.end(), state.observations.begin(),
  		   state.observations.end() );

    // trace the sample if we want to
    if( state.trace_samples ) {
      trace_samples( state, points );
    }
    
    return points;
  }

  //======================================================================

  std::vector<nd_point_t> sample_from_approx_mean
  ( const ruler_point_process_state_t& state )
  {

    scoped_context_switch context( chain_context( context_t( "ruler-point-process-sample" ) ) ) ;
    
    std::vector<nd_point_t> points;
    
    // sample points from each mixture separately
    for( size_t mixture_i = 0; mixture_i < state.mixture_gaussians.size(); ++mixture_i ) {

      gaussian_distribution_t spread_distribution = state.mixture_gaussians[ mixture_i ];
      gaussian_distribution_t start_distribution = state.mixture_ruler_start_gaussians[ mixture_i ];
      gamma_distribution_t period_distribution = state.mixture_period_gammas[ mixture_i ];
      gamma_distribution_t ruler_length_distribution = state.mixture_ruler_length_gammas[ mixture_i ];
      gaussian_distribution_t ruler_direction_distribution = state.mixture_ruler_direction_gaussians[ mixture_i ];
      
      // ok, use means of distributions
      nd_point_t ruler_start = mean( start_distribution );
      nd_direction_t ruler_direction = direction( mean(ruler_direction_distribution) - zero_point( ruler_start.n ) );
      
      // alos use means for period and length
      double period = mean( period_distribution );
      double ruler_length = mean( ruler_length_distribution );
      int num_ticks = (int)floor( ruler_length / period );

      // debug
      if( PRINT_SAMPLE_DEBUG ) {
	std::cout << std::setw(26) << " ";
	std::cout << "{" << mixture_i << "} " << ruler_start << " " << period << " " << ruler_length << " " << ruler_direction << " #ticks: " << num_ticks << std::endl;
      }

      // Ok, now for each point starting at the start of ruler
      // and going for every period, sample a point using hte spread
      for( int tick = 0; tick < num_ticks; ++tick ) {
	
  	// calculate hte tick location
  	nd_point_t tick_point = ruler_start + tick * period * ruler_direction;
	
  	// now sample a variacen about the point
  	nd_point_t p =  sample_from( spread_distribution ) + (tick_point - zero_point(tick_point.n));

	// debug
	if( PRINT_SAMPLE_DEBUG ) {
	  std::cout << std::setw(26) << " " << p << " [" << floor( p.coordinate[0] ) << "]" << std::endl;
	}
	
  	// add to point set
	if( is_inside( p, state.window ) ) {
	  points.push_back( p );
	}
      }
    }


    // sample from a whole new mixture with some probability
    double new_mixture_probability =
      state.model.alpha / 
      ( state.model.alpha + state.mixture_gaussians.size() - 1 );
    if( flip_coin( new_mixture_probability ) ) {
      
      // sample new mixture component from priors
      int dim = state.model.ruler_start_mean_distribution.dimension;
      gaussian_distribution_t zero_gaussian;
      zero_gaussian.dimension = dim;
      zero_gaussian.means = zero_point( dim ).coordinate;
      zero_gaussian.covariance = to_dense_mat( Eigen::MatrixXd::Identity( dim,dim ) * 1e-10 );
      gaussian_distribution_t spread_distribution = 
  	sample_gaussian_from( zero_gaussian,
  			      state.model.precision_distribution );
      gaussian_distribution_t start_distribution = 
  	sample_gaussian_from( state.model.ruler_start_mean_distribution,
  			      state.model.ruler_start_precision_distribution );
      gamma_distribution_t period_distribution = 
  	sample_gamma_from( state.model.period_distribution );
      gamma_distribution_t ruler_length_distribution = 
  	sample_gamma_from( state.model.ruler_length_distribution );
      gaussian_distribution_t ruler_direction_distribution = 
  	sample_gaussian_from( state.model.ruler_direction_mean_distribution,
  			      state.model.ruler_direction_precision_distribution );

      // ok, use means fom distributions
      nd_point_t ruler_start = mean( start_distribution );
      nd_direction_t ruler_direction = direction( mean(ruler_direction_distribution) - zero_point( ruler_start.n ) );
      
      // also use mean for period and a ruler length
      double period = mean( period_distribution );
      double ruler_length = mean( ruler_length_distribution );
      int num_ticks = (int)floor( ruler_length / period );

      // debug
      if( PRINT_SAMPLE_DEBUG ) {
	std::cout << std::setw(26) << " ";
	std::cout << "{NEW} " << ruler_start << " " << period << " " << ruler_length << " " << ruler_direction << " #ticks: " << num_ticks << std::endl;
      }


      // Ok, now for each point starting at the start of ruler
      // and going for every period, sample a point using hte spread
      for( int tick = 0; tick < num_ticks; ++tick ) {
	
  	// calculate hte tick location
  	nd_point_t tick_point = ruler_start + tick * period * ruler_direction;
	
  	// now sample a variacen about the point
  	nd_point_t p = sample_from( spread_distribution ) + (tick_point - zero_point(dim));

	// debug
	if( PRINT_SAMPLE_DEBUG ) {
	  std::cout << std::setw(26) << " " << p << " [" << floor( p.coordinate[0] ) << "]" << std::endl;
	}
	
  	// add to point set
	if( is_inside( p, state.window ) ) {
	  points.push_back( p );
	}
      }
      
    }

    // Ok, now add all of the points already known
    points.insert( points.end(), state.observations.begin(),
  		   state.observations.end() );

    // trace the sample if we want to
    if( state.trace_samples ) {
      trace_samples( state, points );
    }
    
    return points;
  }

  //======================================================================
  //======================================================================
  //======================================================================
  //======================================================================
  //======================================================================
  //======================================================================
  //======================================================================
  //======================================================================

  std::vector<nd_point_t> sample_from
  ( const ruler_point_process_state_t& state )
  {
    // use the approximate mean sampling
    return sample_from_approx_mean( state );
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


}
