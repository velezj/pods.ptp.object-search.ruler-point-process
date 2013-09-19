
#include "mcmc.hpp"
#include <probability-core/distribution_utils.hpp>
#include <point-process-core/point_math.hpp>
#include <point-process-core/context.hpp>
#include <math-core/matrix.hpp>
#include <limits>
#include <math-core/io.hpp>
#include <math-core/utils.hpp>
#include <iostream>
#include <sstream>
#include <fstream>
#include <gsl/gsl_sf_gamma.h>
#include <probability-core/rejection_sampler.hpp>
#include <gsl/gsl_sf_erf.h>
#include <math-core/math_function.hpp>
#include <stdexcept>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_vegas.h>
#include <gsl/gsl_fit.h>
#include <boost/filesystem.hpp>

#define DEBUG_VERBOSE false

namespace ruler_point_process {

  using namespace point_process_core;
  using namespace math_core;
  using namespace probability_core;



  //========================================================================

  // Description:
  // A structu for parameters for monte carlo simulation of marginalization
  // for likelihood computation
  struct likelihood_params_t
  {
    nd_point_t point;
    gaussian_distribution_t spread_distribution;
    gamma_distribution_t period_distribution;
    gamma_distribution_t ruler_length_distribution;
    gaussian_distribution_t ruler_start_distribution;
    gaussian_distribution_t ruler_direction_distribution;
    std::vector<nd_aabox_t> negative_observations;
  };

  // Description:
  // Monte carlco functio nfor the likelihood function
  double likelihood_mc( double* x,
			size_t x_dim,
			void* user_params )
  {
    likelihood_params_t* params = (likelihood_params_t*)user_params;
    int dim = params->point.n;
    assert( 2 * dim + 2 == x_dim );
    int start_slot = 0;
    int dir_slot = start_slot + dim;
    int length_slot = dir_slot + dim;
    int period_slot = length_slot + 1;

    // read in the variables and unpack from X
    nd_point_t ruler_start = zero_point( dim );
    nd_point_t ruler_direction_unnormed = zero_point( dim );
    for( int i = 0; i < dim; ++i ) {
      ruler_start.coordinate[i] = x[start_slot + i];
      ruler_direction_unnormed.coordinate[i] = x[dir_slot + i];
    }
    nd_direction_t ruler_direction = direction( ruler_direction_unnormed - zero_point(dim) );
    double length = x[ length_slot ];
    double period = x[ period_slot ];

    // compute the ticks from the start lengt hand period
    int num_ticks = (int)floor( length / period );

    // debug
    //std::cout << "        .. num ticks: " << num_ticks << " (per: " << period << " len:" << length << ")" << std::endl;

    // Ok, we will now say that is there are too many ticks
    // it's just a nearly impossible thing so low likelhjood
    if( num_ticks > 30 ) {
      return 0;
    }

    // If period is greater than length, zero likelihood
    if( period > length ) {
      return 0;
    }

    // debug
    //std::cout << "  likelihood num ticks: " << num_ticks << std::endl;

    // ok, now just multiple all of the distributions
    // for the point in the params and sum over all ticks
    double lik = 0;
    for( int tick = 0; tick < num_ticks; ++tick ) {
      nd_point_t tick_point = ruler_start + tick * period * ruler_direction;
      nd_point_t p = zero_point(dim) + (params->point - tick_point);
      double this_lik 
	= pdf( p, params->spread_distribution )
	* pdf( ruler_start, params->ruler_start_distribution )
	* pdf( ruler_direction_unnormed, params->ruler_direction_distribution )
	* pdf( length, params->ruler_length_distribution )
	* pdf( period, params->period_distribution );

      // debug
      // std::cout << "   tick: " << tick << " " << tick_point << " orig p: " << params->point << std::endl;
      // std::cout << "      p( p = " << p << " ; " << params->spread_distribution << " ): " << pdf( p, params->spread_distribution ) << std::endl;
      // std::cout << "      p( start = " << ruler_start << " ; " << params->ruler_start_distribution << " ): " << pdf( ruler_start, params->ruler_start_distribution ) << std::endl;
      // std::cout << "      p( dir = " << ruler_direction_unnormed << " ; " << params->ruler_direction_distribution << " ): " << pdf( ruler_direction_unnormed, params->ruler_direction_distribution ) << std::endl;
      // std::cout << "      p( len = " << length << " ; " << params->ruler_length_distribution << " ): " << pdf( length, params->ruler_length_distribution ) << std::endl;
      // std::cout << "      p( period = " << period << " ; " << params->period_distribution << " ): " << pdf( period, params->period_distribution ) << std::endl;
      
      // take care of the negative regions by computing the mass which
      // lies inside and taking that out
      for( size_t i = 0; i < params->negative_observations.size(); ++i ) {

	// need to shift region to be from the tick's point of view
	nd_aabox_t reg = params->negative_observations[i];
	reg.start = zero_point( dim ) + ( reg.start - tick_point );
	reg.end = zero_point( dim ) + ( reg.end - tick_point );

	// compute probability mass inside region and then
	// take the mass outside of region
	double outside_mass = 
	  ( 1.0 - 
	    (cdf(reg.end,params->spread_distribution)
	     - cdf(reg.start,params->spread_distribution )));
	this_lik *= outside_mass;

	// debug
	//std::cout << "      neg region: " << reg << " outside mass: " << outside_mass << std::endl;
      }

      // debug
      //std::cout << "      this lik: " << this_lik << std::endl;

      lik += this_lik;
    }

    // debug
    //std::cout << "  -- total lik: " << lik << std::endl;
    
    return lik;
  }


  // Description:
  // the likelihood ofr a mixture
  double likelihood_of_single_point_for_mixture
  ( const nd_point_t& point,
    const std::vector<nd_aabox_t>& negative_observations,
    const gaussian_distribution_t& spread_distribution,
    const gamma_distribution_t& period_distribution,
    const gamma_distribution_t& ruler_length_distribution,
    const gaussian_distribution_t& ruler_start_distribution,
    const gaussian_distribution_t& ruler_direction_distribution)
  {

    scoped_context_switch context( chain_context( context_t("ruler-process-likelihood") ) );
    
    // this is a really ugly marginalization, so we will use 
    // monte carlo integration to get an estimate 

    // the ordering of the variables:
    // ruler start locaiton (dim slots)
    // ruler direction (dim slots)
    // ruler length (1 slot)
    // ruler period (1 slot)
    int dim = point.n;
    int start_slot = 0;
    int dir_slot = start_slot + dim;
    int length_slot = dir_slot + dim;
    int period_slot = length_slot + 1;

    // First, define the ranges for all of the variables we are
    // marginalizing over
    int num_integrated_vars = dim * 2 + 1 + 1;
    double* ranges_low = new double[ num_integrated_vars ];
    double* ranges_high = new double[ num_integrated_vars ];
    
    // the range of the gaussian to integrate over
    // also used for gammas
    double sigma_range = 3;
    
    // compute the ranges for the start location
    double start_sigma = sqrt(ruler_start_distribution.covariance.data[0]);
    for( int i = 0; i < dim; ++i ) {
      ranges_low[start_slot + i] = ruler_start_distribution.means[i] - sigma_range * start_sigma;
      ranges_high[start_slot + i] = ruler_start_distribution.means[i] + sigma_range * start_sigma;
    }
    
    // compute ranges for direction
    double dir_sigma = sqrt(ruler_direction_distribution.covariance.data[0]);
    for( int i = 0; i < dim; ++i ) {
      ranges_low[dir_slot + i] = ruler_direction_distribution.means[i] - sigma_range * dir_sigma;
      ranges_high[dir_slot + i] = ruler_direction_distribution.means[i] + sigma_range * dir_sigma;
    }

    // compute ranges for period and range
    double period_sigma = sqrt(variance(period_distribution));
    double length_sigma = sqrt(variance(ruler_length_distribution));
    ranges_low[ period_slot ] = mean(period_distribution) - sigma_range * period_sigma;
    if( ranges_low[ period_slot ] < 1e-5 ) {
      ranges_low[ period_slot ] = 1e-5;
    }
    ranges_high[ period_slot ] = mean(period_distribution) + sigma_range * period_sigma;
    if( ranges_high[ period_slot ] > 1e5 ) {
      ranges_high[ period_slot ] = std::max( ranges_low[period_slot], 1e5 );
    }
    ranges_low[ length_slot ] = mean(ruler_length_distribution) - sigma_range * length_sigma;
    if( ranges_low[ length_slot ] < 1e-5 ) {
      ranges_low[ length_slot ] = 1e-5;
    }
    ranges_high[ length_slot ] = mean(ruler_length_distribution) + sigma_range * length_sigma;
    if( ranges_high[ length_slot ] > 1e5 ) {
      ranges_high[ length_slot ] = std::max( ranges_low[length_slot], 1e5 );
    }

 

    // create the gls monte calrlo function
    likelihood_params_t params = { point, 
				   spread_distribution,
				   period_distribution,
				   ruler_length_distribution,
				   ruler_start_distribution,
				   ruler_direction_distribution,
				   negative_observations };
    gsl_monte_function F = { likelihood_mc, num_integrated_vars, &params };
    
    // the number of calls to the monte calro samples
    size_t num_samples = 10;
    size_t warmup_samples = 10;
    size_t num_tries = 0;
    size_t max_tries = 3;
    
    // create the monte carlo and call
    double estimated_lik, estimated_lik_err;
    gsl_monte_vegas_state *s = gsl_monte_vegas_alloc( num_integrated_vars );
    gsl_monte_vegas_integrate( &F, ranges_low, ranges_high, num_integrated_vars,
			       warmup_samples, global_rng(), s,
			       &estimated_lik, &estimated_lik_err );
    do {

      gsl_monte_vegas_integrate( &F, ranges_low, ranges_high, num_integrated_vars,
				 num_samples, global_rng(), s,
				 &estimated_lik, &estimated_lik_err );
      ++num_tries;
      //std::cout << "        converging on likelihood monte-carlo integaration..." << estimated_lik << " (err:" << estimated_lik_err << " chisq:" << gsl_monte_vegas_chisq(s) << ")" << std::endl;
    } while( fabs( gsl_monte_vegas_chisq(s) - 1.0 ) > 0.5 &&
	     num_tries < max_tries);

    // free resources
    gsl_monte_vegas_free( s );
    delete[] ranges_low;
    delete[] ranges_high;

    // fix zero estaimtes
    if( estimated_lik == 0 ) {
      estimated_lik = 1e-20;
      //std::cout << "   vegas mc got 0 volume, setting to " << estimated_lik << std::endl;
    }

    // debug
    //std::cout << "vegas mc tires: " << num_tries << std::endl;

    // return the estiamte  form monte carlo integration
    return estimated_lik;
  }


  //========================================================================


  // Description:
  // the APPROXIMATE likelihood ofr a mixture
  // we jsut take the mean of all the distributions rather than actually
  // computing the tru marginals
  double likelihood_of_single_point_for_mixture_mean_approx
  ( const nd_point_t& point,
    const std::vector<nd_aabox_t>& negative_observations,
    const gaussian_distribution_t& spread_distribution,
    const gamma_distribution_t& period_distribution,
    const gamma_distribution_t& ruler_length_distribution,
    const gaussian_distribution_t& ruler_start_distribution,
    const gaussian_distribution_t& ruler_direction_distribution)
  {

    scoped_context_switch context( chain_context( context_t("ruler-process-likelihood-mean-approx") ) );
    
    
    // this is a really ugly marginalization, so we will use 
    // the means to get an approximation

    // the ordering of the variables:
    // ruler start locaiton (dim slots)
    // ruler direction (dim slots)
    // ruler length (1 slot)
    // ruler period (1 slot)
    int dim = point.n;
    int start_slot = 0;
    int dir_slot = start_slot + dim;
    int length_slot = dir_slot + dim;
    int period_slot = length_slot + 1;
    int num_slots = period_slot + 1;

    // create the gls monte calrlo function
    likelihood_params_t params = { point, 
				   spread_distribution,
				   period_distribution,
				   ruler_length_distribution,
				   ruler_start_distribution,
				   ruler_direction_distribution,
				   negative_observations };
    
    // ok, now ask only for the likelihood given the means of the
    // distributions
    nd_point_t mean_start = mean( ruler_start_distribution );
    nd_point_t mean_dir = mean( ruler_direction_distribution );
    double mean_length = mean( ruler_length_distribution );
    double mean_period = mean( period_distribution );
    double* x = new double[ num_slots ];
    for( int i = 0 ; i < mean_start.n; ++i ) {
      x[ start_slot + i ] = mean_start.coordinate[i];
    }
    for( int i = 0 ; i < mean_dir.n; ++i ) {
      x[ dir_slot + i ] = mean_dir.coordinate[i];
    }
    x[ length_slot ] = mean_length;
    x[ period_slot ] = mean_period;

    double approx_lik = likelihood_mc( x, num_slots, &params );
    

    // return the estiamte  form monte carlo integration
    return approx_lik;
  }


  // //========================================================================

  // nd_point_t
  // resample_mixture_tick_gaussian_mean( const std::vector<nd_point_t>& points, 
  // 				       const std::vector<nd_aabox_t>& negative_observations,
  // 				       const dense_matrix_t& covariance,
  // 				       const poisson_distribution_t& num_distribution,
  // 				       const gaussian_distribution_t& prior )
  // {

  //   // create a new posterior function
  //   boost::shared_ptr<gaussian_mixture_mean_posterior_t>
  //     posterior( new gaussian_mixture_mean_posterior_t 
  // 		 ( points,
  // 		   negative_observations,
  // 		   covariance,
  // 		   num_distribution,
  // 		   prior ) );


  //   // set window to posterior with only points mean
  //   nd_aabox_t window;
  //   window.n = prior.dimension;
  //   window.start = point(posterior->posterior_for_points_only.means);
  //   window.end = point(posterior->posterior_for_points_only.means);

  //   // now extend the window by at least 3 * standard_deviation
  //   double stddev = sqrt(posterior->posterior_for_points_only.covariance.data[0]);
  //   double spread = 3 + negative_observations.size();
  //   if( spread > 5 )
  //     spread = 5;
  //   for( size_t k = 0; k < window.n; ++k ) {
  //     window.start.coordinate[k] -= spread * stddev;
  //     window.end.coordinate[k] += spread * stddev;
  //   }


    
  //   // now rejection sample from this posterior
  //   rejection_sampler_status_t status;
  //   nd_point_t m = 
  //     scaled_rejection_sample<nd_point_t>
  //     ( solid_function<nd_point_t,double>
  // 	(boost::shared_ptr<math_function_t<nd_point_t,double> >(posterior)),
  // 	posterior->scale,
  // 	uniform_point_sampler_within_window( window ),
  // 	status);

  //   return m;
  // }


  //========================================================================
  
  dense_matrix_t
  resample_mixture_tick_gaussian_covariance( const std::vector<nd_point_t>& points,
  					     const std::vector<nd_aabox_t>& negative_observations,
  					     const nd_point_t& mixture_mean,
  					     const gamma_distribution_t& prior)
  {

    size_t dim = mixture_mean.n;
    if( points.empty() == false ) {
      dim = points[0].n;
    }
    if( dim < 1 ) {
      throw std::domain_error("Cannot handle dimension of size < 1" );
    }

    // compute the sum distance sqaured between points and mean
    double sum_distance = 0;
    for( size_t i = 0; i < points.size(); ++i ) {
      sum_distance += distance_sq( points[i], mixture_mean );
    }

    // create the new distributions
    gamma_distribution_t new_dist;
    new_dist.shape = prior.shape + points.size() / 2.0;
    //new_dist.rate = 1.0 / ( sum_distance + prior.shape * prior.rate );
    new_dist.rate = prior.rate + sum_distance / 2.0;
    
    // sample a new precision
    double prec = sample_from(new_dist);
    if( prec == 0 )
      prec = 0.0000000001;
    

    // caclulate hte covariance
    double cov = 1.0 / prec;
    if( cov <= 1e-10 ) {
      cov = 1e-10;
    }
    if( isnan(cov) ) {
      // assume becasue of divide by 0
      cov = 1.0 / 1e-10; 
    }
    
    // returns a covariance matrix which is diagonal with given
    // precision for all elements
    return to_dense_mat( Eigen::MatrixXd::Identity( dim, dim ) * cov );
  }

  //========================================================================
  
  // double
  // resample_mixture_poisson_lambda( const std::vector<nd_point_t>& points,
  // 				   const gamma_distribution_t& prior)
  // {
  //   // new gamma
  //   gamma_distribution_t new_dist;
  //   new_dist.shape = prior.shape + points.size();
  //   new_dist.rate = prior.rate + 1;

  //   return sample_from( new_dist );
  // }



  //========================================================================
  
  nd_point_t
  resample_mixture_ruler_start_gaussian_mean
  ( const std::vector<nd_point_t>& points,
    const std::vector<nd_aabox_t>& negative_observations,
    const dense_matrix_t& covariance,
    const gaussian_distribution_t& prior )
  {
    // create new gaussian for the mean
    // and then samplke from it

    // caculate ruler start as the closer point to the origin
    nd_point_t start;
    int dim = prior.dimension;
    if( points.empty() ) {
      start = sample_from(prior);
    } else {
      start = points[0];
      double min_dist = distance( start, zero_point(dim) );
      for( size_t i = 0; i < points.size(); ++i ) {
	double d = distance( points[i], zero_point(dim) );
	if( d < min_dist ) {
	  min_dist = d;
	  start = points[i];
	}
      }
    }
    
    
    gaussian_distribution_t mean_distribution;
    mean_distribution.dimension = prior.dimension;
    Eigen::VectorXd x = to_eigen_mat( start - zero_point(dim) );
    Eigen::MatrixXd prior_cov = to_eigen_mat( prior.covariance );
    Eigen::VectorXd prior_mean = to_eigen_mat( point( prior.means ) - zero_point(dim) );
    Eigen::MatrixXd cov = to_eigen_mat( covariance );
    Eigen::VectorXd m = ( prior_cov.inverse() + cov.inverse() ).inverse() 
      * ( prior_cov.inverse() * prior_mean + cov.inverse() * x );
    Eigen::MatrixXd post_cov = ( prior_cov.inverse() + cov.inverse() ).inverse();
    mean_distribution.dimension = dim;
    mean_distribution.means = to_vector( m ).component;
    mean_distribution.covariance = to_dense_mat( post_cov );
    
    return sample_from( mean_distribution );
  }


  //========================================================================

  dense_matrix_t
  resample_mixture_ruler_start_gaussian_covariance
  ( const std::vector<nd_point_t>& points,
    const std::vector<nd_aabox_t>& negative_observations,
    const nd_point_t& mean,
    const gamma_distribution_t& prior )
  {
    int dim = mean.n;

    // choose the start
    nd_point_t start;
    if( points.empty() ) {
      start = mean;
    } else {
      start = points[0];
      double min_dist = distance( start, zero_point(dim) );
      for( size_t i = 0; i < points.size(); ++i ) {
	double d = distance( points[i], zero_point(dim) );
	if( d < min_dist ) {
	  min_dist = d;
	  start = points[i];
	}
      }
    }

    // compute the sum distance sqaured between points and mean
    double sum_distance = distance_sq( start, mean );

    // create the new distributions
    gamma_distribution_t new_dist;
    new_dist.shape = prior.shape + 1 / 2.0;
    new_dist.rate = prior.rate + sum_distance / 2.0;
    
    // sample a new precision
    double prec = sample_from(new_dist);
    if( prec == 0 )
      prec = 0.0000000001;
    

    // caclulate hte covariance
    double cov = 1.0 / prec;
    if( cov <= 1e-10 ) {
      cov = 1e-10;
    }
    if( isnan(cov) ) {
      // assume becasue of divide by 0
      cov = 1.0 / 1e-10; 
    }
    
    // returns a covariance matrix which is diagonal with given
    // precision for all elements
    return to_dense_mat( Eigen::MatrixXd::Identity( dim, dim ) * cov );
  }

  //========================================================================
  
  nd_point_t
  resample_mixture_ruler_direction_gaussian_mean
  ( const std::vector<nd_point_t>& points,
    const std::vector<nd_aabox_t>& negative_observations,
    const line_model_t& line,
    const dense_matrix_t& covariance,
    const gaussian_distribution_t& prior )
  {
    // rutnr line into a point
    int dim = prior.dimension;
    if( dim > 2 ) {
      throw std::domain_error( "Cannot handle more than 2D lines!" );
    }
    nd_point_t dir = zero_point(dim);
    dir.coordinate[0] = cos( line.slope );
    if( dim > 1 ) {
      dir.coordinate[1] = sin( line.slope );
    }
    
    gaussian_distribution_t mean_distribution;
    mean_distribution.dimension = prior.dimension;
    Eigen::VectorXd x = to_eigen_mat( dir - zero_point(dim) );
    Eigen::MatrixXd prior_cov = to_eigen_mat( prior.covariance );
    Eigen::VectorXd prior_mean = to_eigen_mat( point( prior.means ) - zero_point(dim) );
    Eigen::MatrixXd cov = to_eigen_mat( covariance );
    Eigen::VectorXd m = ( prior_cov.inverse() + cov.inverse() ).inverse() 
      * ( prior_cov.inverse() * prior_mean + cov.inverse() * x );
    Eigen::MatrixXd post_cov = ( prior_cov.inverse() + cov.inverse() ).inverse();
    mean_distribution.dimension = dim;
    mean_distribution.means = to_vector( m ).component;
    mean_distribution.covariance = to_dense_mat( post_cov );
    
    return sample_from( mean_distribution );
  }


  //========================================================================

  dense_matrix_t
  resample_mixture_ruler_direction_gaussian_covariance
  ( const std::vector<nd_point_t>& points,
    const std::vector<nd_aabox_t>& negative_observations,
    const line_model_t& line,
    const nd_point_t& mean,
    const gamma_distribution_t& prior )
  {
    // rutnr line into a point
    int dim = mean.n;
    if( dim > 2 ) {
      throw std::domain_error( "Cannot handle more than 2D lines!" );
    }
    nd_point_t dir = zero_point(dim);
    dir.coordinate[0] = cos( line.slope );
    if( dim > 1 ) {
      dir.coordinate[1] = sin( line.slope );
    }

    // compute the sum distance sqaured between points and mean
    double sum_distance = distance_sq( dir, mean );

    // create the new distributions
    gamma_distribution_t new_dist;
    new_dist.shape = prior.shape + 1 / 2.0;
    new_dist.rate = prior.rate + sum_distance / 2.0;
    
    // sample a new precision
    double prec = sample_from(new_dist);
    if( prec == 0 )
      prec = 0.0000000001;
    

    // caclulate hte covariance
    double cov = 1.0 / prec;
    if( cov <= 1e-10 ) {
      cov = 1e-10;
    }
    if( isnan(cov) ) {
      // assume becasue of divide by 0
      cov = 1.0 / 1e-10; 
    }
    
    // returns a covariance matrix which is diagonal with given
    // precision for all elements
    return to_dense_mat( Eigen::MatrixXd::Identity( dim, dim ) * cov );
  }
  
  //========================================================================
  
  gamma_distribution_t
  resample_mixture_ruler_period_gamma
  ( const std::vector<nd_point_t>& points,
    const std::vector<nd_aabox_t>& negative_observations,
    const double& period,
    const gamma_conjugate_prior_t& prior )
  {

    gamma_conjugate_prior_t posterior;
    posterior.p = prior.p * period;
    posterior.q = prior.q + period;
    posterior.r = prior.r + 1;
    posterior.s = prior.s + 1;

    // make sure p is never 0, just really small
    if( posterior.p <= 0 ) {
      posterior.p = 1e-10;
    }

    // sample from the posterior
    gamma_distribution_t sample
      = sample_from( posterior );

    // debug
    if( DEBUG_VERBOSE ) {
      std::cout << "    resampling ruler period" << std::endl
		<< "      period: " << period << std::endl
		<< "      prior: " << prior << std::endl
		<< "      posterior: " << posterior << std::endl
		<< "      sample: " << sample 
		<< " [mean: " << mean(sample) 
		<< " var: " << variance(sample) << "] " << std::endl;
    }
    
    // return the sample
    return sample;
  }
  
  //========================================================================
  
  gamma_distribution_t
  resample_mixture_ruler_length_gamma
  ( const std::vector<nd_point_t>& points,
    const std::vector<nd_aabox_t>& negative_observations,
    const double& length,
    const gamma_conjugate_prior_t& prior )
  {

    gamma_conjugate_prior_t posterior;
    posterior.p = prior.p * length;
    posterior.q = prior.q + length;
    posterior.r = prior.r + 1;
    posterior.s = prior.s + 1;

    // make sure p is never 0, just really small
    if( posterior.p <= 0 ) {
      posterior.p = 1e-10;
    }

    // sample from the posterior
    gamma_distribution_t sample 
      = sample_from( posterior );
    
    // debug
    if( DEBUG_VERBOSE ) {
      std::cout << "    resampling ruler length" << std::endl
		<< "      length: " << length << std::endl
		<< "      prior: " << prior << std::endl
		<< "      posterior: " << posterior << std::endl
		<< "      sample: " << sample 
		<< " [mean: " << mean(sample) 
		<< " var: " << variance(sample) << "] " << std::endl;
    }
    
    
    // return the sample
    return sample;
  }
  
  
  //========================================================================
  
  //========================================================================
  
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  
  //========================================================================
  
  // Description:
  // Get a line fomr a set of points
  line_model_t
  fit_line( const std::vector<nd_point_t>& points_arg )
  {
    line_model_t line;
     if( points_arg.empty() )
      throw std::domain_error( "cannot fit line to 0 poitns!" );

     // special case for a single point, assume horizontal line
     // this is a HACK
     if( points_arg.size() == 1 ) {
       line.slope = 0;
       line.intercept = 0;
       return line;
     }

    // Ok,  this only works for 2D points, but we can hack togetehr 1D poitn
    // too by setting thewir y coordiante to 0
    int dim = points_arg[0].n;
    std::vector<nd_point_t> points;
    if( dim > 2 ) {
      throw std::domain_error( "Can only fit a line to 2D or 1D points!" );
    }
    if( dim == 2 ) {
      points = points_arg;
    } else if( dim == 1 ) {
      for( size_t i = 0; i < points_arg.size(); ++i ) {
	points.push_back( point( points_arg[i].coordinate[0], 0 ) );
      }
    }

    // get raw ppints
    double* raw = new double[ points.size() * 2 ];
    for( size_t i = 0; i < points.size(); ++i ) {
      raw[ 2*i ] = points[i].coordinate[0];
      raw[ 2*i + 1 ] = points[i].coordinate[1];
    }

    // use the gsl linear fit funciton
    double m, b;
    double c00, c01, c11; // covariance of fit
    double residual_sum_sq;
    int err = gsl_fit_linear( raw, 2, raw+1, 2, points.size(), &b, &m, &c00, &c01, &c11, &residual_sum_sq);

    // throw exception upong error
    if( err ) {
      throw std::runtime_error( "Error fitting linear regresion line to points using gsl_fit_linear" );
    }
    
    // free resources
    delete[] raw;
 
    // set the line and reutrn it
    line.slope = b;
    line.intercept = m;
    return line;
  }


  //========================================================================

  // Description:
  // Projectpoint onto line
  nd_point_t
  project_onto_line( const nd_point_t& p_arg,
		     const line_model_t& line )
  {
    assert( p_arg.n <= 2 );
    if( p_arg.n > 2 )
      throw std::domain_error( "Can only project 1D or 2D poitnd onto a line");
    nd_point_t p;
    if( p_arg.n == 1 )
      p = point( p_arg.coordinate[0], 0 );
    else
      p = p_arg;

    // special case for infinity slope
    if( line.slope == INFINITY ) {
      return point( line.intercept, p.coordinate[1] );
    }

    nd_point_t p1 = point( 0, line.intercept );
    nd_point_t p2 = point( 1, line.slope + line.intercept );
    nd_vector_t e = p2 - p1;
    nd_vector_t temp = p - p1;
    double d = dot( e, temp );
    temp = e * ( d / magnitude_sq( e ) );
    nd_point_t proj_p = p1 + temp;
    return proj_p;
  }


  //========================================================================

  // Description:
  // Compute the period and length for a line
  void
  compute_period_and_length( const std::vector<nd_point_t>& points,
			     const line_model_t& line,
			     const gamma_distribution_t& period_prior,
			     const gamma_distribution_t& length_prior,
			     double& period,
			     double& length)
  {

    if( points.empty() ) {
      throw std::domain_error( "Unable to compute period and length from 0 points!" );
    }
    
    // special case of 1 point
    if( points.size() == 1 ) {
      period = sample_from( period_prior );
      length = sample_from( length_prior );
      return;
    }

    // copy points and sort them
    std::vector<nd_point_t> sorted_points( points.begin(),
					   points.end() );

    // debug
    if( false ) {
      for( size_t i = 0; i < sorted_points.size(); ++i ) {
	if( sorted_points[i].n < 1 || sorted_points[i].n > 2 ) {
	  std::cout << "%% sorted_points[" << i << "].n \\NotIn [1,2]" << std::endl;
	}
      }
    }
    
    std::sort( sorted_points.begin(),
	       sorted_points.end(),
	       point_lexicographical_compare );
    
    // OK, now compute the mean difference
    std::vector<double> diffs;
    for( size_t i = 0; i < sorted_points.size() -1 ; ++i ) {
      diffs.push_back( distance( sorted_points[i], sorted_points[i+1] ) );
    }
    
    // make period be mean difference
    period = mean( diffs );
    
    // make the length be the distance btween first and last points
    length = distance( sorted_points[0], sorted_points[sorted_points.size()-1] );
  }

  //========================================================================

  // Description:
  // Calculate the ruler specifications given a set of points
  // As in the line, length, and period
  void
  calculate_best_fit_ruler_params
  ( std::vector<nd_point_t>& points,
    const gamma_distribution_t& period_prior,
    const gamma_distribution_t& length_prior,
    line_model_t& line,
    double& period,
    double& length )
  {
    
    // fit the line
    line = fit_line( points );
   
    // ok, now compute the period and length
    compute_period_and_length( points, 
			       line,
			       period_prior,
			       length_prior,
			       period, length );
    
  }

  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  
  //========================================================================
  
  //========================================================================
  
  //========================================================================
  
  
  //========================================================================
  
  class alpha_posterior_likelihood_t
  {
  public:
    alpha_posterior_likelihood_t( double a,
  				  double num_mix,
  				  double num_ob )
      : alpha(a),
  	num_mixtures( num_mix ),
  	num_obs( num_ob )
    {}
    
    double alpha;
    double num_mixtures;
    double num_obs;
    
    double operator()( const double& x ) const
    {
      return gsl_sf_gamma( x ) * pow(x, (double)num_mixtures - 2 ) * exp( - 1.0 / x ) / gsl_sf_gamma( x + num_obs );
    }
  };
  
  double 
  resample_alpha_hyperparameter( const double& alpha,  
  				 const std::size_t& num_mixtures,
  				 const std::size_t& num_obs )
  {
    alpha_posterior_likelihood_t lik( alpha,
     				      num_mixtures,
     				      num_obs );
    
    rejection_sampler_status_t status;
    double sampled_alpha =
      autoscale_rejection_sample<double>
      (lik, 0.00001, num_obs + 2, status );
    return sampled_alpha;
  }
  
  //========================================================================
  
  // gaussian_distribution_t
  // resample_mean_distribution_hyperparameters( igmm_point_process_state_t& state )
  // {
  //   assert( state.model.mean_distribution.dimension == 1 );
  //   if( state.model.mean_distribution.dimension != 1 ) {
  //     throw std::domain_error( "Only implemented for 1D gaussians!" );
  //   }
    
  //   double previous_precision = 1.0 / state.model.mean_distribution.covariance.data[0];
  //   double hyperprior_precision = 1.0 / state.model.prior_variance;
    
  //   // ok, sum the means of the current mixtures
  //   double mean_sum = 0;
  //   for( std::size_t i = 0; i < state.mixture_gaussians.size(); ++i ) {
  //     mean_sum += state.mixture_gaussians[i].means[0];
  //   }

  //   // compute the new variance of the distribution over means
  //   double new_variance = 
  //     1.0 / ( previous_precision * state.mixture_gaussians.size() + hyperprior_precision );
    
  //   // create the new gaussian for hte mean
  //   gaussian_distribution_t new_mean_dist;
  //   new_mean_dist.dimension = 1;
  //   new_mean_dist.means.push_back( ( previous_precision * mean_sum + state.model.prior_mean * hyperprior_precision ) * new_variance );
  //   new_mean_dist.covariance = to_dense_mat( Eigen::MatrixXd::Identity(1,1) * new_variance );

  //   // sample a new mean
  //   nd_point_t new_mean = sample_from( new_mean_dist );
    
  //   // sum the sqaured ereror to this new mean
  //   // to compute distribution of new precision
  //   double sum_diff = 0;
  //   for( std::size_t i = 0; i < state.mixture_gaussians.size(); ++i ) {
  //     sum_diff += distance_sq( point(state.mixture_gaussians[i].means),
  // 			       new_mean );
  //   }
    
  //   // crea the precision distribution
  //   gamma_distribution_t new_precision_dist;
  //   new_precision_dist.shape = ( state.mixture_gaussians.size() / 2.0 + state.model.precision_distribution.shape );
  //   //new_precision_dist.rate = 1.0 / ( 2 * ( sum_diff + 1.0/hyperprior_precision));
  //   new_precision_dist.rate = sum_diff / 2.0 + state.model.precision_distribution.rate;
    
  //   // sample a new precision
  //   double new_precision = sample_from( new_precision_dist );
    
  //   // return a new gaussian
  //   gaussian_distribution_t sampled_mean_dist;
  //   sampled_mean_dist.dimension = new_mean_dist.dimension;
  //   sampled_mean_dist.means = new_mean.coordinate;
  //   sampled_mean_dist.covariance = to_dense_mat( Eigen::MatrixXd::Identity(1,1) * 1.0 / new_precision );

  //   return sampled_mean_dist;
  
  // }
  
  //========================================================================
  
// Description:
  // The explicit posterior form for the shape of a gamma
  // given some precisions as data.
  // This is used for sampling a posterior gamma prior
  // for a precision variable ( so for example, if you
  // have a distribution of precision as a gamma, then 
  // this will givne you the posterior of the shape paramter 
  // of this gamma given some observed precicions )
  //
  // This is *NOT* to be used unless you know what you are doing,
  // there is a better API for this using hte 
  // sample_precision_gamma_prior function which internally uses this!
  class precision_shape_posterior_t
  {
  public:

    // Description:
    // This is the "factor" of the posterior likelihood
    // that is because of the observed precisions.
    // This is, formally:
    //    given precision observations s_1 ... s_k
    //    given rate w
    //    given previous/prior shape B
    //  factor = s_1 ^ (B/2) * e^(-B * w * s_1 / 2) * .... * s_k ^ (B/2) * e^(-B * w * s_k / 2)
    //
    // TODO: this is WRONG, the B should be the true posterior B, 
    //       hence factor cannot be precomputed!!!
    double factor;

    // Descripiton:
    // The number of observed precision used to compute the factor above
    double k;

    // Descripion:
    // The rate of the gamma we are resampling,
    // or just the rate of the gamma we are getting hte posterior
    // shape parameter over
    double rate;
    
    precision_shape_posterior_t( double factor, double k,
				 double rate )
      : factor(factor),
  	k(k),
  	rate(rate)
    {}
    
    double operator() (double b) const
    {
      double h = gsl_sf_gamma( b/2.0 );
      if( h > 1000 )
  	return 0.0;
      if( h < 0.00001 )
  	return 0.0;
      double r = 1.0 / pow( h, k );
      r *= pow( b * rate / 2.0, (k * b - 3.0) / 2.0 );
      r *= exp( - 1.0 / ( 2.0 * b ) );
      r *= factor;
      if( r > 10000 )
  	return 0;
      return r;
    }
  };
  
  gamma_distribution_t
  resample_precision_distribution_hyperparameters( ruler_point_process_state_t& state )
  {
    assert( state.model.ruler_start_mean_distribution.dimension == 1 );
    if( state.model.ruler_start_mean_distribution.dimension != 1 ) {
      throw std::domain_error( "Only implemented for 1D points!" );
    }
    
    double b = state.model.precision_distribution.shape;
    double w = state.model.precision_distribution.rate;
    
    // sum the precisions of each mixture
    // as well as the factor for them
    double prec_sum = 0;
    double prec_factor = 1;
    for( std::size_t i = 0; i < state.mixture_gaussians.size(); ++i ) {
      double prec = ( 1.0 / state.mixture_gaussians[i].covariance.data[0] ); 
      prec_sum += prec;
      prec_factor *= pow( prec, b/2.0) * exp( - b * w * prec / 2.0 );
    }
    
    precision_shape_posterior_t lik(prec_factor,
     				    state.mixture_gaussians.size(),
     				    state.model.precision_distribution.rate );
    
    // Hack, just sample some values and sample from those
    uniform_sampler_within_range uniform( 0.00001, 100 );
    std::vector<double> sample_precs;
    std::vector<double> sample_vals;
    for( std::size_t i = 0; i < 100; ++i ) {
      double x = uniform();
      double l = lik(x);
      sample_vals.push_back( x );
      sample_precs.push_back( l );
    }
    discrete_distribution_t dist;
    dist.n = sample_precs.size();
    dist.prob = sample_precs;
    double new_shape = sample_vals[sample_from(dist)];
    
    
    // now build up the distribution for the rate of the precision
    gamma_distribution_t new_rate_dist;
    new_rate_dist.shape = ( state.mixture_gaussians.size() * new_shape + 1 ) / 2.0;
    new_rate_dist.rate = 2 * 1.0 / ( new_shape * prec_sum + 1.0 / state.model.prior_variance );
    
    
    // sample a new precision rate
    double new_rate = sample_from(new_rate_dist);
    
    // return hte new distribution
    gamma_distribution_t new_dist;
    new_dist.shape = new_shape;
    new_dist.rate = new_rate;
    
    return new_dist;
  }
  
  //========================================================================
  
  // gamma_distribution_t
  // resample_num_points_per_gaussian_distribution_hyperparameters( igmm_point_process_state_t& state )
  // {

  //   size_t sum_num = 0;
  //   for( size_t i = 0; i < state.mixture_gaussians.size(); ++i ) {
  //     sum_num += points_for_mixture( state, i ).size();
  //   }

  //   gamma_distribution_t new_dist;
  //   new_dist.shape = ( state.model.num_points_per_gaussian_distribution.shape + sum_num );
  //   new_dist.rate = state.model.num_points_per_gaussian_distribution.rate + state.mixture_gaussians.size();

  //   return new_dist;
  // }

  //========================================================================

  // Description:
  // The posterior for a gamma conjugate prior likelihood
  // and a poisson prior for a single parameter for the gamma
  // conjugate prior (this is for hyperparameter resampling!)
  class single_parameter_gcp_likelihood_poisson_prior_posterior_t
  {
  public:
    std::vector<gamma_distribution_t> gamma_data_points;
    gamma_conjugate_prior_t base_likelihood_gcp;
    gamma_distribution_t prior;
    char param;
    
    single_parameter_gcp_likelihood_poisson_prior_posterior_t
    ( const std::vector<gamma_distribution_t>& points,
      const gamma_conjugate_prior_t& base_lik,
      const gamma_distribution_t& prior,
      const char param)
      : gamma_data_points(points),
	base_likelihood_gcp( base_lik ),
	prior( prior ),
	param(param)
    {
    }      
    
    double operator() ( const double& x ) const 
    {
      double mult = 1;
      gamma_conjugate_prior_t likelihood_gcp = base_likelihood_gcp;
      switch( param ) {
      case 'p':
	likelihood_gcp.p = x;
	break;
      case 'q':
	likelihood_gcp.q = x;
	break;
      case 'r':
	likelihood_gcp.r = x;
	break;
      case 's':
	likelihood_gcp.s = x;
	break;
      default:
	throw std::domain_error( "Unknown gamma conjugate prior parameter!" );
      }
      
      for( size_t i = 0; i < gamma_data_points.size(); ++i ) {
	mult *= likelihood( gamma_data_points[i], likelihood_gcp );
      }	
      mult *= pdf( x, prior );
      return mult;
    }
  };

  // Descripiton:
  // Sample from the posterior of a gamma conjugate prior likelihood
  // with a poisson prior (because we are non negative )
  double sample_gcp_liklihood_poisson_prior_posterior
  ( const std::vector<gamma_distribution_t>& gamma_data_points,
    const gamma_conjugate_prior_t& lik,
    const gamma_distribution_t& prior,
    const char parameter,
    const double low_arg = -1,
    const double high_arg = -1,
    const int num_samples = 300)
  {
    
    // create a new posterior functor
    single_parameter_gcp_likelihood_poisson_prior_posterior_t
      posterior( gamma_data_points, lik, prior, parameter);

    // fix a range to sample within
    double low = 2;
    double high = 100;

    // use given arguments 
    if( low_arg != -1 )
      low = low_arg;
    if( high_arg != -1 )
      high = high_arg;

    // the range is based on the actual parameter
    if( low_arg == -1 &&
	high_arg == -1 ) {
      switch( parameter ) {
      case 'p':
	high = 1e3;
	break;
      case 'q':
	low = 1e-3;
	high = 10;
	break;
      case 'r':
	high = num_samples + low;
	break;
      case 's':
	high = num_samples + low;
	break;
      default:
	throw std::domain_error( "Unknown gamma conjugate prior parameter!" );
      }
    }
    
    // rejection sample
    // by first sampling discrete points in range
    std::vector< double > samples;
    std::vector< double > samples_p;
    double step = ( high - low ) / num_samples;
    if( step < 1e-6 )
      step = 1e-6;

    // debug
    if( DEBUG_VERBOSE ) {
      std::cout << "      param " << parameter << ", range [" << low << " " << high << "], step=" << step << std::endl;
    }

    // take the discrete samples, then sample from this discrete distribution
    for( double i = low; i < high; i += step ) {
      double x = i;
      samples.push_back( x );
      double p = posterior( x );
      samples_p.push_back( p );
    }
    int idx = sample_from( discrete_distribution( samples_p ) );
    double sample = samples[ idx ];

    // debug
    if( DEBUG_VERBOSE ) {
      // std::cout << "        s={";
      // for( size_t i = 0; i < samples.size(); ++i ) {
      //   std::cout << "            " << samples[i] << ": " << samples_p[i] << std::endl;
      // }
      // std::cout << "          }" << std::endl;
      std::cout << "        sampled: " << sample << "  p(...) = " << samples_p[idx] << std::endl;
    }
    
    return sample;
  }
  
  //========================================================================
  
  // gamma_conjugate_prior_t
  // resample_period_distribution_hyperparameters
  // ( const ruler_point_process_state_t& state )
  // {
    // // create the gamma priors
    // gamma_distribution_t p_prior;
    // p_prior.shape = state.model.prior_ruler_period_p_shape;
    // p_prior.rate = state.model.prior_ruler_period_p_rate;
    // gamma_distribution_t q_prior;
    // q_prior.shape = state.model.prior_ruler_period_q_shape;
    // q_prior.rate = state.model.prior_ruler_period_q_rate;
    // gamma_distribution_t r_prior;
    // r_prior.shape = state.model.prior_ruler_period_r_shape;
    // r_prior.rate = state.model.prior_ruler_period_r_rate;
    // gamma_distribution_t s_prior;
    // s_prior.shape = state.model.prior_ruler_period_s_shape;
    // s_prior.rate = state.model.prior_ruler_period_s_rate;
    
    // // // create poisson priors
    // // poisson_distribution_t p_prior;
    // // p_prior.lambda = state.model.prior_ruler_period_p_lambda;
    // // poisson_distribution_t q_prior;
    // // q_prior.lambda = state.model.prior_ruler_period_q_lambda;
    // // poisson_distribution_t r_prior;
    // // r_prior.lambda = state.model.prior_ruler_period_r_lambda;
    // // poisson_distribution_t s_prior;
    // // s_prior.lambda = state.model.prior_ruler_period_s_lambda;

    // // debug
    // if( DEBUG_VERBOSE ) {
    //   std::cout << "    hyper period sampling: " << std::endl;

    //   // print out the current statistics
    //   double cur_mean, cur_var;
    //   estimate_gamma_conjugate_prior_sample_stats( state.model.period_distribution, cur_mean, cur_var );
    //   std::cout << "      current mean: " << cur_mean << " var: " << cur_var << std::endl;
      
    //   // print out the gamma observations
    //   std::cout << "      gamma obs: " << std::endl;
    //   for( int i = 0; i < state.mixture_period_gammas.size(); ++i ) {
    // 	gamma_distribution_t g = state.mixture_period_gammas[i];
    // 	std::cout << "        " << g << " [mean: " << mean(g) << " var: " << variance(g) << "]  " << std::endl;
    //   }
    // }

    // // keep track of gcp likelihood as things change
    // gamma_conjugate_prior_t lik = state.model.period_distribution;
    
    // // sample each parameter individually
    // double p = sample_gcp_liklihood_poisson_prior_posterior
    //   ( state.mixture_period_gammas,
    // 	lik,
    // 	p_prior,
    // 	'p',
    // 	0.001,
    // 	40);
    // lik.p = p;
    // double q = sample_gcp_liklihood_poisson_prior_posterior
    //   ( state.mixture_period_gammas,
    // 	lik,
    // 	q_prior,
    // 	'q',
    // 	0.001,
    // 	30);
    // lik.q = q;
    // double r = sample_gcp_liklihood_poisson_prior_posterior
    //   ( state.mixture_period_gammas,
    // 	lik,
    // 	r_prior,
    // 	'r');
    // lik.r = r;
    // double s = sample_gcp_liklihood_poisson_prior_posterior
    //   ( state.mixture_period_gammas,
    // 	lik,
    // 	s_prior,
    // 	's');
    // lik.s = s;

    // gamma_conjugate_prior_t sample;
    // sample.p = p;
    // sample.q = q;
    // sample.r = r;
    // sample.s = s;

    // // HACK: always set the r,s to be the right numbers
    // //sample.r = state.observations.size();
    // //sample.s = state.observations.size();

    // // HACK: always set s to be what r was
    // sample.s = sample.r;

    // // debug
    // if( DEBUG_VERBOSE ) {
    //   std::cout << "    hyper period sampled: " << sample << std::endl;
      
    //   // Compute an estiamte of the mean for the hyper parameter
    //   // and of the variance
    //   double mean, var;
    //   estimate_gamma_conjugate_prior_sample_stats( sample, mean, var );
    //   std::cout << "      mean: " << mean << " var: " << var << std::endl;
    // }

    // return sample;
    // }

  gamma_conjugate_prior_t
  resample_period_distribution_hyperparameters
  ( const ruler_point_process_state_t& state )
  {

    // debug
    //std::cout << "    hyper period: " << std::endl;
    
    // Compute the posterior given the *actual* periods of the 
    // mixtures in the state
    gamma_conjugate_prior_t posterior
      = state.model.period_distribution;
    for( size_t i = 0; i < state.mixture_period_gammas.size(); ++i ) {
      
      // calculate the actual length of this mixture
      std::vector<nd_point_t> points = points_for_mixture( state, i );
      double period = 1;
      double length = 1;
      line_model_t line = { 1, 0 };
      calculate_best_fit_ruler_params( points, 
				       state.mixture_period_gammas[ i ],
				       state.mixture_ruler_length_gammas[ i ],
				       line,
				       period, 
				       length );
      double x = period;
      posterior.p *= x;
      posterior.q += x;
      posterior.r += 1;
      posterior.s += 1;
      
      // debug
      //std::cout << "       obs x: " << x << std::endl;
      
    }
    
    // debug
    //std::cout << "      prior:     " << state.model.period_distribution << std::endl;
    //std::cout << "      posterior: " << posterior << std::endl;
    
    return fix_numerical_reduce_counts( posterior );
  }
  
  //========================================================================
  
  // gamma_conjugate_prior_t
  // resample_ruler_length_hyperparameters
  // ( const ruler_point_process_state_t& state )
  // {
  //   // create the gamma priors
  //   gamma_distribution_t p_prior;
  //   p_prior.shape = state.model.prior_ruler_length_p_shape;
  //   p_prior.rate = state.model.prior_ruler_length_p_rate;
  //   gamma_distribution_t q_prior;
  //   q_prior.shape = state.model.prior_ruler_length_q_shape;
  //   q_prior.rate = state.model.prior_ruler_length_q_rate;
  //   gamma_distribution_t r_prior;
  //   r_prior.shape = state.model.prior_ruler_length_r_shape;
  //   r_prior.rate = state.model.prior_ruler_length_r_rate;
  //   gamma_distribution_t s_prior;
  //   s_prior.shape = state.model.prior_ruler_length_s_shape;
  //   s_prior.rate = state.model.prior_ruler_length_s_rate;


  //   // // create poisson priors
  //   // poisson_distribution_t p_prior;
  //   // p_prior.lambda = state.model.prior_ruler_length_p_lambda;
  //   // poisson_distribution_t q_prior;
  //   // q_prior.lambda = state.model.prior_ruler_length_q_lambda;
  //   // poisson_distribution_t r_prior;
  //   // r_prior.lambda = state.model.prior_ruler_length_r_lambda;
  //   // poisson_distribution_t s_prior;
  //   // s_prior.lambda = state.model.prior_ruler_length_s_lambda;

  //   // debug
  //   if( DEBUG_VERBOSE ) {
  //     std::cout << "    hyper length sampling: " << std::endl;
      
  //     // print out the current statistics
  //     double cur_mean, cur_var;
  //     estimate_gamma_conjugate_prior_sample_stats( state.model.ruler_length_distribution, cur_mean, cur_var );
  //     std::cout << "      current mean: " << cur_mean << " var: " << cur_var << std::endl;
      
  //     // print out the gamma observations
  //     std::cout << "      gamma obs: " << std::endl;
  //     for( int i = 0; i < state.mixture_ruler_length_gammas.size(); ++i ) {
  // 	gamma_distribution_t g = state.mixture_ruler_length_gammas[i];
  // 	std::cout << "        " << g << " [mean: " << mean(g) << " var: " << variance(g) << "]  " << std::endl;
  //     }
  //   }

  //   // keep track of gcp likelihood as things change
  //   gamma_conjugate_prior_t lik = state.model.ruler_length_distribution;
    
  //   // sample each parameter individually
  //   double p = sample_gcp_liklihood_poisson_prior_posterior
  //     ( state.mixture_ruler_length_gammas,
  // 	lik,
  // 	p_prior,
  // 	'p',
  // 	1,
  // 	pow( 10 , (state.model.ruler_length_distribution.r+1) ) + 50,
  // 	1000);
  //   lik.p = p;
  //   double q = sample_gcp_liklihood_poisson_prior_posterior
  //     ( state.mixture_ruler_length_gammas,
  // 	lik,
  // 	q_prior,
  // 	'q',
  // 	0.001,
  // 	10 * (state.observations.size() + 2));
  //   lik.q = q;
  //   double r = sample_gcp_liklihood_poisson_prior_posterior
  //     ( state.mixture_ruler_length_gammas,
  // 	lik,
  // 	r_prior,
  // 	'r');
  //   lik.r = r;
  //   double s = sample_gcp_liklihood_poisson_prior_posterior
  //     ( state.mixture_ruler_length_gammas,
  // 	lik,
  // 	s_prior,
  // 	's');
  //   lik.s = s;

  //   gamma_conjugate_prior_t sample;
  //   sample.p = p;
  //   sample.q = q;
  //   sample.r = r;
  //   sample.s = s;

  //   // HACK: always set the r,s to be the right numbers
  //   //sample.r = state.observations.size();
  //   //sample.s = state.observations.size();

  //   // HACK: always set s to be what r was
  //   sample.s = sample.r;
    
  //   // debug
  //   if( DEBUG_VERBOSE ) {
  //     std::cout << "    hyper length sampled: " << sample << std::endl;
      
  //     // Compute an estiamte of the mean for the hyper parameter
  //     // and of the variance
  //     double mean, var;
  //     estimate_gamma_conjugate_prior_sample_stats( sample, mean, var );
  //     std::cout << "      mean: " << mean << " var: " << var << std::endl;
  //   }


  //   return sample;
  // }


  gamma_conjugate_prior_t
  resample_ruler_length_hyperparameters
  ( const ruler_point_process_state_t& state )
  {

    // debug
    //std::cout << "    hyper length: " << std::endl;
  
    // ok, just compute posterior given the mixture gammas
    // for a few samples and choose one at random
    std::vector<gamma_conjugate_prior_t> posterior_samples;
    std::vector<double> posterior_liks;
    int num_posterior_samples = 1;
    for( int k = 0; k < num_posterior_samples; ++k ) {

      // calculate a new posterior by sampling a point from every mixture 
      // gamma and updating hte prior
      // Hack: we are just going to use the mean from the gammas
      //double mean_x = 0;
      gamma_conjugate_prior_t posterior
	= state.model.ruler_length_distribution;
      for( size_t i = 0; i < state.mixture_ruler_length_gammas.size(); ++i ) {
	// mean_x += mean( state.mixture_ruler_length_gammas[i] );
	// double x = sample_from( state.mixture_ruler_length_gammas[i] );

	// calculate the actual length of this mixture
  	std::vector<nd_point_t> points = points_for_mixture( state, i );
	double period = 1;
	double length = 1;
	line_model_t line = { 1, 0 };
	calculate_best_fit_ruler_params( points, 
					 state.mixture_period_gammas[ i ],
					 state.mixture_ruler_length_gammas[ i ],
					 line,
					 period, 
					 length );
	double x = length;
	posterior.p *= x;
	posterior.q += x;
	posterior.r += 1;
	posterior.s += 1;

	// debug
	//std::cout << "       obs x: " << x << std::endl;

      }
      // mean_x /= state.mixture_ruler_length_gammas.size();
      // posterior.p *= mean_x;
      // posterior.q += mean_x;
      // posterior.r += 1;
      // posterior.s += 1;
      posterior_samples.push_back( posterior );

      // calculate the likelihood of all the gammas in mixture
      // independently
      double lik = 1;
      for( size_t i = 0 ; i < state.mixture_ruler_length_gammas.size(); ++i ) {
	lik *= likelihood( state.mixture_ruler_length_gammas[i], posterior );
      }
      posterior_liks.push_back( lik );

      // debug
      //std::cout << "      prior:     " << state.model.ruler_length_distribution << std::endl;
      //std::cout << "      posterior: " << posterior << " (lik: " << lik << ")" << std::endl;
    }

    // Sample posterior according to likelihood
    int idx = sample_from( discrete_distribution( posterior_liks ) );
    gamma_conjugate_prior_t sample
      = posterior_samples[ idx ];

    return fix_numerical_reduce_counts( sample );
  }
  
  
  //========================================================================
  
  gaussian_distribution_t
  resample_ruler_start_mean_hyperparameters
  ( const ruler_point_process_state_t& state )
  {
    
    // construct verctor of observations as the mixutre means
    std::vector<nd_point_t> observed_means;
    for( size_t i = 0; i < state.mixture_ruler_start_gaussians.size(); ++i ) {
      observed_means.push_back( point( state.mixture_ruler_start_gaussians[i].means ) );
    }

    // we treat the covariance as a single variance over all dimensions
    double current_variance = state.model.ruler_start_mean_distribution.covariance.data[0];

    // debug
    if( DEBUG_VERBOSE ) {
      std::cout << "  Resample ruler start mean: " << std::endl;
    }
    
    // now sample a new gaussian mean prior
    // treating the covariance as a single variance with an Identity matrix!
    gaussian_distribution_t sample
      = sample_mean_gaussian_prior
      ( observed_means,
	current_variance,
	state.model.ruler_start_precision_distribution,
	state.model.prior_ruler_start_mean,
	state.model.prior_ruler_start_variance );

    return sample;
  }
  
  //========================================================================
  
  gamma_distribution_t
  resample_ruler_start_precision_hyperparameters
  ( const ruler_point_process_state_t& state )
  {
    
    // get the precicins for the ruler starts as observations
    // Treate each mixture as having a single covariance element
    std::vector<double> observed_prec;
    for( size_t i = 0; i < state.mixture_ruler_start_gaussians.size(); ++i ) {
      observed_prec.push_back( 1.0 / state.mixture_ruler_start_gaussians[i].covariance.data[0] );
    }

    // debug
    if( DEBUG_VERBOSE ) {
      std::cout << "  Resample ruler start precision: " << std::endl;
    }


    // sample a new precision gamma prior
    gamma_distribution_t sample
      = sample_precision_gamma_prior
      ( observed_prec,
	state.model.ruler_start_precision_distribution,
	state.model.prior_ruler_start_variance );
    
    return sample;
  }
  
  //========================================================================
  
  gaussian_distribution_t
  resample_ruler_direction_mean_hyperparameters
  ( const ruler_point_process_state_t& state )
  {
    // construct verctor of observations as the mixutre means
    std::vector<nd_point_t> observed_means;
    for( size_t i = 0; i < state.mixture_ruler_direction_gaussians.size(); ++i ) {
      observed_means.push_back( point( state.mixture_ruler_direction_gaussians[i].means ) );
    }

    // we treat the covariance as a single variance over all dimensions
    double current_variance = state.model.ruler_direction_mean_distribution.covariance.data[0];

    // debug
    if( DEBUG_VERBOSE ) {
      std::cout << "  Resample ruler direction mean: " << std::endl;
    }

    
    // now sample a new gaussian mean prior
    // treating the covariance as a single variance with an Identity matrix!
    gaussian_distribution_t sample
      = sample_mean_gaussian_prior
      ( observed_means,
	current_variance,
	state.model.ruler_direction_precision_distribution,
	state.model.prior_ruler_direction_mean,
	state.model.prior_ruler_direction_variance );

    return sample;
  }
  
  //========================================================================
  
  gamma_distribution_t
  resample_ruler_direction_precision_hyperparameters
  ( const ruler_point_process_state_t& state )
  {
    // get the precicins for the ruler starts as observations
    // Treate each mixture as having a single covariance element
    std::vector<double> observed_prec;
    for( size_t i = 0; i < state.mixture_ruler_direction_gaussians.size(); ++i ) {
      observed_prec.push_back( 1.0 / state.mixture_ruler_direction_gaussians[i].covariance.data[0] );
    }

    // debug
    if( DEBUG_VERBOSE ) {
      std::cout << "  Resample ruler direction precision: " << std::endl;
    }


    // sample a new precision gamma prior
    gamma_distribution_t sample
      = sample_precision_gamma_prior
      ( observed_prec,
	state.model.ruler_direction_precision_distribution,
	state.model.prior_ruler_direction_variance );
    
    return sample;
  }
  
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================

  void trace_mcmc( const ruler_point_process_state_t& state )
  {
    std::ostringstream oss;
    oss << state.trace_mcmc_dir << "/" << context_filename( "mcmc.trace" );
    //oss << state.trace_mcmc_dir << "/" << "mcmc.trace";
    std::string filename = oss.str();
    boost::filesystem::create_directories( boost::filesystem::path( filename ).parent_path() );
    std::ofstream fout_trace( filename.c_str(), std::ios_base::app | std::ios_base::out );
    fout_trace 
      << state.iteration << " "
      << state.model.alpha << " "
      << state.model.precision_distribution.shape << " "
      << state.model.precision_distribution.rate << " "
      << state.model.period_distribution.p << " "
      << state.model.period_distribution.q << " "
      << state.model.period_distribution.r << " "
      << state.model.period_distribution.s << " ";
    double m, var;
    estimate_gamma_conjugate_prior_sample_stats
      ( state.model.period_distribution, m, var );
    fout_trace
      << m << " " 
      << var << " "
      << state.model.ruler_length_distribution.p << " "
      << state.model.ruler_length_distribution.q << " "
      << state.model.ruler_length_distribution.r << " "
      << state.model.ruler_length_distribution.s << " ";
    estimate_gamma_conjugate_prior_sample_stats
      ( state.model.ruler_length_distribution, m, var );
    fout_trace
      << m << " "
      << var << " "
      << state.model.ruler_start_mean_distribution.means[0] << " "
      << state.model.ruler_start_mean_distribution.covariance.data[0] << " "
      << state.model.ruler_start_precision_distribution.shape << " "
      << state.model.ruler_start_precision_distribution.rate << " "
      << state.model.ruler_direction_mean_distribution.means[0] << " "
      << state.model.ruler_direction_mean_distribution.covariance.data[0] << " "
      << state.model.ruler_direction_precision_distribution.shape << " "
      << state.model.ruler_direction_precision_distribution.rate << " "
      << state.observations.size() << " ";
    for( size_t i = 0; i < state.observation_to_mixture.size(); ++i ) {
      fout_trace << state.observation_to_mixture[i] << " ";
    }
    fout_trace << state.mixture_gaussians.size() << " ";
    for( size_t i = 0; i < state.mixture_gaussians.size(); ++i ) {
      fout_trace << state.mixture_gaussians[i].means[0] << " "
		 << state.mixture_gaussians[i].covariance.data[0] << " "
		 << state.mixture_period_gammas[i].shape << " " 
		 << state.mixture_period_gammas[i].rate << " "
		 << mean(state.mixture_period_gammas[i]) << " "
		 << variance( state.mixture_period_gammas[i]) << " "
		 << state.mixture_ruler_length_gammas[i].shape << " "
		 << state.mixture_ruler_length_gammas[i].rate << " "
		 << mean( state.mixture_ruler_length_gammas[i] ) << " "
		 << variance( state.mixture_ruler_length_gammas[i] ) << " "
		 << state.mixture_ruler_start_gaussians[i].means[0] << " "
		 << state.mixture_ruler_start_gaussians[i].covariance.data[0] << " "
		 << state.mixture_ruler_direction_gaussians[i].means[0] << " "
		 << state.mixture_ruler_direction_gaussians[i].covariance.data[0] << " ";
    }
    fout_trace << std::endl;
    fout_trace.flush();
  }

  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  
  
  void mcmc_single_step( ruler_point_process_state_t& state )
  {

    scoped_context_switch context( chain_context( context_t("ruler-process-mcmc") ) ); 
    
    // For each observation, sample a corresponding mixture for it
    // from the known mixtures (with their parameters) as well as a
    // completely new mixture (using the DP-alpha parameter)

    for( size_t observation_i = 0; observation_i < state.observations.size(); ++observation_i ) {

      // debug
      // std::cout << "OBSERVATION " << observation_i << std::endl;

      // remove mixture it this observation is the only observation in it

      size_t old_mixture_index = state.observation_to_mixture[ observation_i ];
      if( state.mixture_to_observation_indices[ state.observation_to_mixture[ observation_i ] ].size() == 1 ) {


  	remove( state.mixture_gaussians, old_mixture_index );
  	remove( state.mixture_period_gammas, old_mixture_index );
  	remove( state.mixture_ruler_length_gammas, old_mixture_index );
  	remove( state.mixture_ruler_start_gaussians, old_mixture_index );
  	remove( state.mixture_ruler_direction_gaussians, old_mixture_index );
  	remove( state.mixture_to_observation_indices, old_mixture_index );

  	// fix obs->mixture mapping since we now have one less mixture
  	for( size_t i = 0; i < state.observation_to_mixture.size(); ++i ) {
  	  if( i == observation_i ) {
  	    state.observation_to_mixture[i] = std::numeric_limits<size_t>::max();
  	  } else if( state.observation_to_mixture[i] > old_mixture_index ) {
	    	    
  	    state.observation_to_mixture[i] -= 1;
  	  }
  	}

	
  	// set the old index to be the MAX since we just removed the
  	// entire mixture (so ther eis no old mixture left)
  	old_mixture_index = std::numeric_limits<size_t>::max();

      } 

      // compute the likelihood for each known cluster, and multiply
      // by the number of poitns in that cluster to get the likelihood
      // of the observation belonging to that cluster
      std::vector<double> likelihoods = std::vector<double>();
      for( size_t mixture_i = 0; mixture_i < state.mixture_gaussians.size(); ++mixture_i ) {
	
  	// number of point belonging to mixture (discounting ourselves)
  	size_t num_obs_in_mixture = state.mixture_to_observation_indices[ mixture_i ].size();
  	if( state.observation_to_mixture[ observation_i ] == mixture_i ) {
  	  --num_obs_in_mixture;
  	}

	// debug
	// std::cout << "  MIXTURE: " << mixture_i << " (obs" << observation_i << ")" << std::endl;

  	// likelihood of this observation comming from this mixtiure
  	double lik = 
  	  likelihood_of_single_point_for_mixture
  	  ( state.observations[ observation_i ],
	    state.negative_observations,
  	    state.mixture_gaussians[ mixture_i ],
  	    state.mixture_period_gammas[ mixture_i ],
  	    state.mixture_ruler_length_gammas[ mixture_i ],
  	    state.mixture_ruler_start_gaussians[ mixture_i ],
  	    state.mixture_ruler_direction_gaussians[ mixture_i ]);
  	lik *= ( num_obs_in_mixture /
  		 ( state.observations.size() -1 + state.model.alpha ) );

	// debug
	if( DEBUG_VERBOSE ) {
	  std::cout << "  mixture: " << mixture_i << " obs: " << observation_i << " " << state.observations[ observation_i ] << std::endl;
	  std::cout << "    alpha:  " << state.model.alpha << std::endl;
	  std::cout << "    spread: " << state.mixture_gaussians[ mixture_i ] << std::endl;
	  std::cout << "    period: " << state.mixture_period_gammas[ mixture_i ] << std::endl;
	  std::cout << "    length: " << state.mixture_ruler_length_gammas[ mixture_i ] << std::endl;
	  std::cout << "    start : " << state.mixture_ruler_start_gaussians[ mixture_i ] << std::endl;
	  std::cout << "    direct: " << state.mixture_ruler_direction_gaussians[ mixture_i ] << std::endl;
	  std::cout << "    Num Obs: " << num_obs_in_mixture << " lik: " << lik << std::endl;
	}

	
  	// store this likelihood (to later sample from it)
  	likelihoods.push_back( lik );
      }

      // add the likelihood of a complete new mixture
      // here we use a SINGLE sample for the expected 
      // new model from our hyperparameters
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
      double new_mixture_lik = 
  	likelihood_of_single_point_for_mixture
  	( state.observations[ observation_i ],
	  state.negative_observations,
  	  spread_distribution,
  	  period_distribution,
  	  ruler_length_distribution,
  	  start_distribution,
  	  ruler_direction_distribution );
      new_mixture_lik *= ( state.model.alpha /
  			   ( state.observations.size() -1 + state.model.alpha ) );
      if( new_mixture_lik < 0 ) {
  	new_mixture_lik = 0;
      }
      likelihoods.push_back( new_mixture_lik );

      // debug
      if( DEBUG_VERBOSE ) {
	std::cout << "  New mixture obs: " << observation_i << " " << state.observations[ observation_i ] << std::endl;
	std::cout << "    alpha:  " << state.model.alpha << std::endl;
	std::cout << "    spread: " << spread_distribution << std::endl;
	std::cout << "    period: " << period_distribution << std::endl;
	std::cout << "    length: " << ruler_length_distribution << std::endl;
	std::cout << "    start : " << start_distribution << std::endl;
	std::cout << "    direct: " << ruler_direction_distribution << std::endl;
	std::cout << "    lik: " << new_mixture_lik << std::endl;
      }


      // Now sample a mixture for this observation from the likelihoods
      size_t new_mixture_index = sample_from( discrete_distribution( likelihoods ) );

      // debug
      if( DEBUG_VERBOSE ) {
	std::cout << "  Sampled Mixture: " << new_mixture_index << " lik: " << likelihoods[ new_mixture_index ] << std::endl;
      }
      
      // if this is a new mixture, sample new parameters for it
      if( new_mixture_index == likelihoods.size() - 1 ) {
  	state.mixture_gaussians.push_back( spread_distribution );
  	state.mixture_period_gammas.push_back( period_distribution );
  	state.mixture_ruler_length_gammas.push_back( ruler_length_distribution );
  	state.mixture_ruler_start_gaussians.push_back( start_distribution );
  	state.mixture_ruler_direction_gaussians.push_back( ruler_direction_distribution );
  	state.mixture_to_observation_indices.push_back( std::vector<size_t>() );
      }
      
      // update correspondances and observation index mappings
      state.observation_to_mixture[ observation_i ] = new_mixture_index;
      state.mixture_to_observation_indices[ new_mixture_index ].push_back( observation_i );
      if( old_mixture_index < state.mixture_to_observation_indices.size() ) {
  	for( size_t i = 0; i < state.mixture_to_observation_indices[ old_mixture_index ].size(); ++i ) {
  	  if( state.mixture_to_observation_indices[ old_mixture_index ][ i ] == observation_i ) {
	    
  	    remove( state.mixture_to_observation_indices[ old_mixture_index ],
  		    i );
	    
  	    break; 
  	    // remove only the FIRST index (since we may have a double because
  	    // we stayed at the same mixture
  	  }
  	}
      }

      // resample the mixture parameters
      for( size_t mixture_i = 0; mixture_i < state.mixture_gaussians.size(); ++mixture_i ) {

	// debug
	if( DEBUG_VERBOSE ) {
	  std::cout << "Resample Observation " << observation_i << " Mixture: " << mixture_i << std::endl;
	}

  	// first get the points in the mixture component
  	std::vector<nd_point_t> points = points_for_mixture( state, mixture_i );
  	std::vector<nd_aabox_t> negative_observations = state.negative_observations;
	double period = 1;
	double length = 1;
	line_model_t line = { 1, 0 };
	calculate_best_fit_ruler_params( points, 
					 state.mixture_period_gammas[ mixture_i ],
					 state.mixture_ruler_length_gammas[ mixture_i ],
					 line,
					 period, 
					 length );

	// debug
	if( DEBUG_VERBOSE ) {
	  std::cout << "  points: ";
	  for( size_t p_i = 0; p_i < points.size(); ++p_i ) {
	    std::cout << points[p_i] << " ";
	  }
	  std::cout << std::endl;
	  std::cout << "  -- Ruler len: " << length << " Period: " << period << " line: " << line.slope << " " << line.intercept << std::endl;
	}

  	// resample the mean of the ticks
  	int dim = state.model.ruler_start_mean_distribution.dimension;
  	nd_point_t mixture_tick_mean = zero_point( dim );

  	// resample the spread of the ticks
  	dense_matrix_t mixture_tick_covariance
  	  = resample_mixture_tick_gaussian_covariance( points,
  						       negative_observations,
  						       mixture_tick_mean,
  						       state.model.precision_distribution );
	
  	// resample the start of the ruler
  	nd_point_t mixture_ruler_start_mean
  	  = resample_mixture_ruler_start_gaussian_mean( points,
  							negative_observations,
							state.mixture_ruler_start_gaussians[ mixture_i ].covariance,
  							state.model.ruler_start_mean_distribution );
  	dense_matrix_t mixture_ruler_start_covariance
  	  = resample_mixture_ruler_start_gaussian_covariance( points,
  							      negative_observations,
  							      mixture_ruler_start_mean,
  							      state.model.ruler_start_precision_distribution );

  	// resample the period of the ruler
  	gamma_distribution_t mixture_ruler_period
  	  = resample_mixture_ruler_period_gamma( points,
  						 negative_observations,
						 period,
  						 state.model.period_distribution );

  	// resample the length of the ruler
  	gamma_distribution_t mixture_ruler_length
  	  = resample_mixture_ruler_length_gamma( points,
  						 negative_observations,
						 length,
  						 state.model.ruler_length_distribution );
	
  	// resample the direction of the ruler
  	nd_point_t mixture_ruler_direction_mean
  	  = resample_mixture_ruler_direction_gaussian_mean
	  ( points,
	    negative_observations,
	    line,
	    state.mixture_ruler_direction_gaussians[ mixture_i ].covariance,
	    state.model.ruler_direction_mean_distribution );
  	dense_matrix_t mixture_ruler_direction_covariance
  	  = resample_mixture_ruler_direction_gaussian_covariance
	  ( points,
	    negative_observations,
	    line,
	    mixture_ruler_direction_mean,
	    state.model.ruler_direction_precision_distribution );
	

  	// set the new mixture parameters
  	state.mixture_gaussians[ mixture_i ].means = mixture_tick_mean.coordinate;
  	state.mixture_gaussians[ mixture_i ].covariance = mixture_tick_covariance;
  	state.mixture_period_gammas[ mixture_i ] = mixture_ruler_period;
  	state.mixture_ruler_start_gaussians[ mixture_i ].means = mixture_ruler_start_mean.coordinate;
  	state.mixture_ruler_start_gaussians[ mixture_i ].covariance = mixture_ruler_start_covariance;
  	state.mixture_ruler_length_gammas[ mixture_i ] = mixture_ruler_length;
  	state.mixture_ruler_direction_gaussians[ mixture_i ].means = mixture_ruler_direction_mean.coordinate;
  	state.mixture_ruler_direction_gaussians[ mixture_i ].covariance = mixture_ruler_direction_covariance;

	// debug
	if( DEBUG_VERBOSE ) {
	  std::cout << "+ Resample Observation " << observation_i << " Mixture: " << mixture_i << " POST updated mixtures: " << std::endl;
	  std::cout << "   tick spread: " << state.mixture_gaussians[ mixture_i ] << std::endl;
	  std::cout << "   period: " << state.mixture_period_gammas[ mixture_i ] << "[mean: " << mean(state.mixture_period_gammas[mixture_i]) << " var: " << variance( state.mixture_period_gammas[ mixture_i ] )<< "]" << std::endl;
	  std::cout << "   length: " << state.mixture_ruler_length_gammas[ mixture_i ] << "[mean: " << mean(state.mixture_ruler_length_gammas[mixture_i]) << " var: " << variance( state.mixture_ruler_length_gammas[ mixture_i ] )<< "]" << std::endl;
	  std::cout << "   start: " << state.mixture_ruler_start_gaussians[ mixture_i ] << std::endl;
	  std::cout << "   direction: " << state.mixture_ruler_direction_gaussians[ mixture_i ] << std::endl;
	}
      }

    }


    // // ok, compute posteriors of the hyperparameter distributions
    
    // debug
    if( DEBUG_VERBOSE ) {
      std::cout << "HYPERPARAMETER update" << std::endl;
    }
    
    // Resample a new alpha
    state.model.alpha 
      = resample_alpha_hyperparameter( state.model.alpha,
				       state.mixture_gaussians.size(),
				       state.observations.size() );
    
    // resample the new precision distribution
    // state.model.precision_distribution
    //   = resample_precision_distribution_hyperparameters( state );
    
    // resample the period hyperparameters
    // state.model.period_distribution
    //   = resample_period_distribution_hyperparameters( state );
    
    // resample ruler length
    // state.model.ruler_length_distribution
    //   = resample_ruler_length_hyperparameters( state );
    
    // resample ruler start
    // state.model.ruler_start_mean_distribution
    //   = resample_ruler_start_mean_hyperparameters( state );
    // state.model.ruler_start_precision_distribution
    //   = resample_ruler_start_precision_hyperparameters( state );
    
    // resample ruler direction
    // state.model.ruler_direction_mean_distribution
    //   = resample_ruler_direction_mean_hyperparameters( state );
    // state.model.ruler_direction_precision_distribution
    //   = resample_ruler_direction_precision_hyperparameters( state );
    

    // save race if we want to
    if( state.trace_mcmc ) {
      trace_mcmc( state );
    }

    // increment mcmc iteration
    state.iteration += 1;
  }
			 

  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================



}
