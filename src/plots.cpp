
#include "plots.hpp"
#include <plot-server/api/plot.hpp>
#include <probability-core/distribution_utils.hpp>


using namespace plot_server;
using namespace plot_server::api;
using namespace boost::property_tree;
using namespace math_core;
using namespace probability_core;

namespace ruler_point_process {

  //===============================================================


  // Description:
  // transform nd_point_t into data_point_t using default
  // 0 -> x, 1 -> y, 2 -> z coordinate mapping
  data_point_t
  nd_point_to_data_point( const nd_point_t& p )
  {

    if( p.n == 1 ) {
      return data_point_t( p.coordinate[0], 
			   0.0, 
			   0.0 );
    } else if( p.n == 2 ) {
      return data_point_t( p.coordinate[0], 
			   p.coordinate[1], 
			   0.0 );
    } else {
      return data_point_t( p.coordinate[0], 
			   p.coordinate[1], 
			   p.coordinate[3] );
    }
  }
    
  //===============================================================

  std::string
  create_window_dataseries( const ruler_point_process_t& pp,
			    const ptree& extra_config,
			    const std::string& title )
  {
    std::vector<data_point_t> data_points;
    nd_aabox_t window = pp.window();
    data_points.push_back( nd_point_to_data_point( window.start ) );
    data_points.push_back
      ( nd_point_to_data_point( point( window.end.coordinate[0],
				       window.start.coordinate[1] ) ) );
    data_points.push_back( nd_point_to_data_point( window.end ) );
    data_points.push_back
      ( nd_point_to_data_point( point( window.start.coordinate[0],
				       window.end.coordinate[1] ) ) );
    data_points.push_back( nd_point_to_data_point( window.start ) );

    ptree config = extra_config;
    return add_data_series( data_points,
			    config,
			    title );
  }

  //===============================================================

  std::string
  create_observations_dataseries( const ruler_point_process_t& pp,
				  const ptree& extra_config,
				  const std::string& title )
  {
    std::vector<data_point_t> data_points;
    for( nd_point_t obs_p : pp.observations() ) {
      data_points.push_back( nd_point_to_data_point( obs_p ) );
    }

    ptree config = extra_config;
    return add_data_series( data_points,
			    config,
			    title );
  }

  //===============================================================

  std::string
  create_mean_tick_dataseries( const ruler_point_process_t& pp,
			       const ptree& extra_config,
			       const std::string& title )
  {
    // we want to get the mean of all of the current ticks and
    // make those ticks into points
    std::vector<data_point_t> data_points;
   
    // look at each mixture separately and get it's ticks
    for( size_t mixture_i = 0; mixture_i < pp._state.mixture_gaussians.size(); ++mixture_i ) {
      
      gaussian_distribution_t spread_distribution = pp._state.mixture_gaussians[ mixture_i ];
      gaussian_distribution_t start_distribution = pp._state.mixture_ruler_start_gaussians[ mixture_i ];
      gamma_distribution_t period_distribution = pp._state.mixture_period_gammas[ mixture_i ];
      gamma_distribution_t ruler_length_distribution = pp._state.mixture_ruler_length_gammas[ mixture_i ];
      gaussian_distribution_t ruler_direction_distribution = pp._state.mixture_ruler_direction_gaussians[ mixture_i ];
      
      // get the means of each distribution
      nd_point_t ruler_start = mean( start_distribution );
      nd_direction_t ruler_direction = direction( mean(ruler_direction_distribution) - zero_point( ruler_start.n ) );
      double period = mean( period_distribution );
      double ruler_length = mean( ruler_length_distribution );
      int num_ticks = (int)floor( ruler_length / period );

      //std::cout << title << ":  " << "ticks(" << mixture_i << ") = " << num_ticks << std::endl;
      //std::cout << title << ":  " << "period: " << period << " length: " << ruler_length << std::endl;
      //std::cout << title << ":  " << "period: " << period_distribution << " length: " << ruler_length_distribution << std::endl;

      // add the tick points
      for( int tick = 0; tick < num_ticks; ++tick ) {
	
  	// calculate hte tick location
  	nd_point_t tick_point = ruler_start + tick * period * ruler_direction;
	
	// add to data_points
	data_points.push_back( nd_point_to_data_point( tick_point ) );
      }
    }

    // create the data series
    ptree config = extra_config;
    return add_data_series( data_points,
			    config,
			    title );
  }

  //===============================================================

  std::string
  create_gaussian_mixture_ellipse_dataseries
  ( const ruler_point_process_t& pp,
    const ptree& extra_config,
    const std::string& title,
    const double& at_sigma = 2.0)
  {

    // compute the center of the ticks (like above)
    // and the compute the major,minor nad angle for the 
    // ellipse wanted
    // we want to get the mean of all of the current ticks and
    // make those ticks into points
    std::vector<data_point_t> data_points;
   
    // look at each mixture separately and get it's ticks
    for( size_t mixture_i = 0; mixture_i < pp._state.mixture_gaussians.size(); ++mixture_i ) {
      
      gaussian_distribution_t spread_distribution = pp._state.mixture_gaussians[ mixture_i ];
      gaussian_distribution_t start_distribution = pp._state.mixture_ruler_start_gaussians[ mixture_i ];
      gamma_distribution_t period_distribution = pp._state.mixture_period_gammas[ mixture_i ];
      gamma_distribution_t ruler_length_distribution = pp._state.mixture_ruler_length_gammas[ mixture_i ];
      gaussian_distribution_t ruler_direction_distribution = pp._state.mixture_ruler_direction_gaussians[ mixture_i ];
      
      // get the means of each distribution
      nd_point_t ruler_start = mean( start_distribution );
      nd_direction_t ruler_direction = direction( mean(ruler_direction_distribution)  - zero_point( ruler_start.n ) );
      double period = mean( period_distribution );
      double ruler_length = mean( ruler_length_distribution );
      int num_ticks = (int)floor( ruler_length / period );

      // calculate the ellipse for the spread distribution
      //double ellipse_radius = at_sigma * sqrt(variance( spread_distribution ));
      double ellipse_radius = at_sigma * sqrt( spread_distribution.covariance.data[0] );

      // add the tick points
      for( int tick = 0; tick < num_ticks; ++tick ) {
	
  	// calculate hte tick location
  	nd_point_t tick_point = ruler_start + tick * period * ruler_direction;

	// create a datapoint with the tick's x or x,y
	// and the radius as the third variable
	data_point_t dp = nd_point_to_data_point( tick_point );
	dp.attributes.put( "major_diameter", 2.0 * ellipse_radius );
	dp.attributes.put( "minor_diameter", 2.0 * ellipse_radius );
	dp.attributes.put( "angle", 0.0 );
	data_points.push_back( dp );
      }
    }

    // create the data series
    ptree config = extra_config;
    return add_data_series( data_points,
			    config,
			    title );

  }

  //===============================================================

  std::string
  create_mean_ruler_line_dataseries
  ( const ruler_point_process_t& pp,
    const ptree& extra_config,
    const std::string& title )
  {
    // this will have the dataseries be a set of start (x,y) and
    // a set of dx,dy additions since it assumes we will
    // plot with the "vectors" style
    std::vector<data_point_t> data_points;
   
    // look at each mixture separately and get it's mean ruler
    for( size_t mixture_i = 0; mixture_i < pp._state.mixture_gaussians.size(); ++mixture_i ) {
      
      gaussian_distribution_t start_distribution = pp._state.mixture_ruler_start_gaussians[ mixture_i ];
      gamma_distribution_t ruler_length_distribution = pp._state.mixture_ruler_length_gammas[ mixture_i ];
      gaussian_distribution_t ruler_direction_distribution = pp._state.mixture_ruler_direction_gaussians[ mixture_i ];
      
      // get the means of each distribution
      nd_point_t ruler_start = mean( start_distribution );
      nd_point_t ruler_direction = mean(ruler_direction_distribution);
      double mag = magnitude( ruler_direction - zero_point(ruler_start.n) );
      for( int c = 0; c < ruler_direction.n; ++c ) {
	ruler_direction.coordinate[c] /= mag;
      }
      double ruler_length = mean( ruler_length_distribution );

      // add the start +length at direction vectors
      data_point_t dp = nd_point_to_data_point( ruler_start );
      dp.attributes.put( "dx", ruler_length * ruler_direction.coordinate[0] );
      if( ruler_direction.n > 1 ) {
	dp.attributes.put( "dy", ruler_length * ruler_direction.coordinate[1] );
      } else {
	dp.attributes.put( "dy", 0.0 );
      }
      data_points.push_back( dp );
      
    }

    // create the data series
    ptree config = extra_config;
    return add_data_series( data_points,
			    config,
			    title );
    
  }

  //===============================================================

  std::string
  create_ruler_period_distribution_dataseries
  ( const ruler_point_process_t& pp,
    const ptree& extra_config,
    const std::string& title )
  {
    // compute the start of the rulers.
    // Then we want to draw some lines which are samples of the
    // period distribution from the *start* of the ruler
    std::vector<data_point_t> data_points;

    size_t num_gamma_samples = 100;
    double sigma_spread = 3.0;
   
    // look at each mixture separately and get it's ticks
    for( size_t mixture_i = 0; mixture_i < pp._state.mixture_gaussians.size(); ++mixture_i ) {
      
      gaussian_distribution_t start_distribution = pp._state.mixture_ruler_start_gaussians[ mixture_i ];
      gamma_distribution_t period_distribution = pp._state.mixture_period_gammas[ mixture_i ];
      gamma_distribution_t ruler_length_distribution = pp._state.mixture_ruler_length_gammas[ mixture_i ];
      gaussian_distribution_t ruler_direction_distribution = pp._state.mixture_ruler_direction_gaussians[ mixture_i ];
      
      // get the means of each distribution
      nd_point_t ruler_start = mean( start_distribution );
      nd_direction_t ruler_direction = direction( mean(ruler_direction_distribution) - zero_point( ruler_start.n ) );
      double mean_period = mean( period_distribution );
      double ruler_length = mean( ruler_length_distribution );
      int num_ticks = (int)floor( ruler_length / mean_period );

      // calculate the direction normal to the line
      nd_direction_t normal_direction = 
	direction( 1.0 / magnitude( vector( ruler_direction ) ) 
		   * vector( -ruler_direction.value[1],
			     ruler_direction.value[0] ) );
      
      // get a coarse representation of the period gamma to draw
      double var_period = variance( period_distribution );
      double min_period = mean_period - sigma_spread * sqrt( var_period );
      if ( min_period < 1e-5 ) {
	min_period = 1e-5;
      }
      double max_period = mean_period + sigma_spread * sqrt( var_period );
      double step = ( max_period - min_period ) / num_gamma_samples;
      std::vector<double> period_samples;
      std::vector<double> period_p;
      double max_p = 0;
      for( double x = min_period; x < max_period; x += step ) {
	period_samples.push_back( x );
	double p = pdf( x, period_distribution );
	period_p.push_back( p );
	if( max_p < p ) {
	  max_p = p;
	}
      }
      // make the likelihoods have a maximum of a known scale
      // since we are drawing these
      double wanted_max_p = 1.0 / 20 * magnitude( pp.window().end -
						  pp.window().start );
      for( int i = 0; i < period_p.size(); ++i ) {
	period_p[ i ] = period_p[ i ] / max_p * wanted_max_p;
      }

      // add the tick points
      for( int tick = 0; tick < num_ticks; ++tick ) {
	
  	// for eavery sampled period, add a sampled gamma distribution
	// starting at the start for the n'th "tick"
	for( int si = 0; si < period_samples.size(); ++si ) {
	  double period = period_samples[ si ];
	  double p = period_p[ si ];
	  nd_point_t tick_point = ruler_start + tick * period * ruler_direction + p * normal_direction;
	  
	  data_points.push_back( nd_point_to_data_point( tick_point ) );
	}

      }

      // add a brak to the dataseries
      data_point_t dp_skip = data_point_t( ptree() );
      dp_skip.attributes.put( "skip", true );
      data_points.push_back( dp_skip );
    }

    // create the data series
    ptree config = extra_config;
    return add_data_series( data_points,
			    config,
			    title );

    
  }

  //===============================================================

  std::string
  create_ruler_length_distribution_dataseries
  ( const ruler_point_process_t& pp,
    const ptree& extra_config,
    const std::string& title )
  {
    // compute the start of the rulers.
    // Then we want to draw some lines which are samples of the
    // end distribution from the mean start
    std::vector<data_point_t> data_points;

    size_t num_gamma_samples = 100;
    double sigma_spread = 3.0;
   
    // look at each mixture separately and get it's ticks
    for( size_t mixture_i = 0; mixture_i < pp._state.mixture_gaussians.size(); ++mixture_i ) {
      
      gaussian_distribution_t start_distribution = pp._state.mixture_ruler_start_gaussians[ mixture_i ];
      gamma_distribution_t ruler_length_distribution = pp._state.mixture_ruler_length_gammas[ mixture_i ];
      gaussian_distribution_t ruler_direction_distribution = pp._state.mixture_ruler_direction_gaussians[ mixture_i ];
      
      // get the means of each distribution
      nd_point_t ruler_start = mean( start_distribution );
      nd_direction_t ruler_direction = direction( mean(ruler_direction_distribution) - zero_point( ruler_start.n ) );
      double mean_length = mean( ruler_length_distribution );

      // calculate the direction normal to the line
      nd_direction_t normal_direction = 
	direction( 1.0 / magnitude( vector( ruler_direction ) ) 
		   * vector( -ruler_direction.value[1],
			     ruler_direction.value[0] ) );
      
      // get a coarse representation of the length gamma to draw
      double var_length = variance( ruler_length_distribution );
      double min_length = mean_length - sigma_spread * sqrt( var_length );
      if( min_length < 1e-5 ) {
	min_length = 1e-5;
      }
      double max_length = mean_length + sigma_spread * sqrt( var_length );
      double step = ( max_length - min_length ) / num_gamma_samples;
      std::vector<double> length_samples;
      std::vector<double> length_p;
      double max_p = 0;
      for( double x = min_length; x < max_length; x += step ) {
	length_samples.push_back( x );
	double p = pdf( x, ruler_length_distribution );
	length_p.push_back( p );
	if( max_p < p ) {
	  max_p = p;
	}
      }
      
      // make the likelihoods have a maximum of a known scale
      // since we are drawing these
      double wanted_max_p = 1.0 / 20 * magnitude( pp.window().end -
						  pp.window().start );
      for( int i = 0; i < length_p.size(); ++i ) {
	length_p[ i ] = length_p[ i ] / max_p * wanted_max_p;
      }
	
      // for eavery sampled period, add a sampled gamma distribution
      // starting at the start
      for( int si = 0; si < length_samples.size(); ++si ) {
	double length = length_samples[ si ];
	double p = length_p[ si ];
	nd_point_t end_point = ruler_start + length * ruler_direction + p * normal_direction;
	
	data_points.push_back( nd_point_to_data_point( end_point ) );

      }

      // add a break to the dataseries
      data_point_t dp_skip = data_point_t( ptree() );
      dp_skip.attributes.put( "skip", true );
      data_points.push_back( dp_skip );
    }

    // create the data series
    ptree config = extra_config;
    return add_data_series( data_points,
			    config,
			    title );

    
  }

  //===============================================================

  std::string
  create_ruler_start_distribution_dataseries
  ( const ruler_point_process_t& pp,
    const ptree& extra_config,
    const std::string& title,
    const double& at_sigma = 2.0)
  {

    // compute the center of the ticks (like above)
    // and the compute the major,minor nad angle for the 
    // ellipse wanted
    // we want to get the mean of all of the current ticks and
    // make those ticks into points
    std::vector<data_point_t> data_points;
   
    // look at each mixture separately and get it's ticks
    for( size_t mixture_i = 0; mixture_i < pp._state.mixture_gaussians.size(); ++mixture_i ) {
      
      gaussian_distribution_t spread_distribution = pp._state.mixture_gaussians[ mixture_i ];
      gaussian_distribution_t start_distribution = pp._state.mixture_ruler_start_gaussians[ mixture_i ];
      gamma_distribution_t period_distribution = pp._state.mixture_period_gammas[ mixture_i ];
      gamma_distribution_t ruler_length_distribution = pp._state.mixture_ruler_length_gammas[ mixture_i ];
      gaussian_distribution_t ruler_direction_distribution = pp._state.mixture_ruler_direction_gaussians[ mixture_i ];
      
      // get the means of each distribution
      nd_point_t ruler_start = mean( start_distribution );
      double period = mean( period_distribution );
      double ruler_length = mean( ruler_length_distribution );
      int num_ticks = (int)floor( ruler_length / period );

      // calculate the ellipse for the spread distribution
      //double ellipse_radius = at_sigma * sqrt(variance( start_distribution ));
      double ellipse_radius = at_sigma * sqrt( start_distribution.covariance.data[0] );

      // add the start point ellipse
      nd_point_t tick_point = ruler_start;

      // create a datapoint with the tick's x or x,y
      // and the radius as the third variable
      data_point_t dp = nd_point_to_data_point( tick_point );
      dp.attributes.put( "major_diameter", 2.0 * ellipse_radius );
      dp.attributes.put( "minor_diameter", 2.0 * ellipse_radius );
      dp.attributes.put( "angle", 0.0 );
      data_points.push_back( dp );
    }

    // create the data series
    ptree config = extra_config;
    return add_data_series( data_points,
			    config,
			    title );

  }


  //===============================================================

  nd_point_t
  project_onto_line( const nd_point_t& p_arg,
		     const nd_point_t& line_p,
		     const nd_direction_t& line_dir)
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
    if( line_dir.value[0] == 0 ) {
      return point( line_p.coordinate[0], p.coordinate[1] );
    }

    nd_point_t p1 = line_p;
    nd_point_t p2 = line_p + vector( line_dir );
    nd_vector_t e = p2 - p1;
    nd_vector_t temp = p - p1;
    double d = dot( e, temp );
    temp = e * ( d / magnitude_sq( e ) );
    nd_point_t proj_p = p1 + temp;
    return proj_p;
  }

  //===============================================================

  std::string
  create_observation_projection_dataseries
  ( const ruler_point_process_t& pp,
    const ptree& extra_config,
    const std::string& title )
  {
    std::vector<data_point_t> data_points;
   
    // look at each mixture separately and get it's mean ruler
    for( size_t mixture_i = 0; mixture_i < pp._state.mixture_gaussians.size(); ++mixture_i ) {
      
      gaussian_distribution_t start_distribution = pp._state.mixture_ruler_start_gaussians[ mixture_i ];
      gamma_distribution_t ruler_length_distribution = pp._state.mixture_ruler_length_gammas[ mixture_i ];
      gaussian_distribution_t ruler_direction_distribution = pp._state.mixture_ruler_direction_gaussians[ mixture_i ];
      
      // get the means of each distribution
      nd_point_t ruler_start = mean( start_distribution );
      nd_direction_t ruler_direction = direction( mean(ruler_direction_distribution) - zero_point( ruler_start.n ) );
      double ruler_length = mean( ruler_length_distribution );

      // ok, now we get the observation points that belong to this
      // fuler
      std::vector<nd_point_t> obs_for_mixture
	= points_for_mixture( pp._state, mixture_i );
      
      // for each observation, calculate the projection line from the
      // observation onto the ruler
      for( nd_point_t obs : obs_for_mixture ) {
	nd_point_t proj_obs = project_onto_line( obs,
						 ruler_start,
						 ruler_direction );
	data_point_t s = nd_point_to_data_point( obs );
	data_point_t e = nd_point_to_data_point( proj_obs );
	data_point_t sk = data_point_t( ptree() );
	sk.attributes.put( "skip", true );
	data_points.push_back( s );
	data_points.push_back( e );
	data_points.push_back( sk );
      }      
    }

    // create the data series
    ptree config = extra_config;
    return add_data_series( data_points,
			    config,
			    title );
    
  }

  //===============================================================
  //===============================================================
  //===============================================================
  //===============================================================
  //===============================================================
  //===============================================================
  //===============================================================
  //===============================================================
  //===============================================================
  //===============================================================
  //===============================================================
  
  //===============================================================

  std::string
  plot_ruler_point_process( const ruler_point_process_t& pp,
			    const std::string& title) {

    // define a data series with the point process window
    std::string window_ds
      = create_window_dataseries
      ( pp,
	ptree(),
	"window-of-" + title );

    // define a data series with the observations for the point process_t
    std::string obs_ds
      = create_observations_dataseries
      ( pp,
	ptree(),
	"observations-of-" + title );
    
    // define a data series with the mean *ticks* for the
    // ruler mixtures
    std::string mean_tick_ds
      = create_mean_tick_dataseries
      ( pp,
	ptree(),
	"mean-mixtures-of-" + title );
    
    // Define a data series with the given stddiv ellipses of
    // the mixture gaussians
    std::string gaussian_mixture_ellipse_ds
      = create_gaussian_mixture_ellipse_dataseries
      ( pp,
	ptree(),
	"covariance-ellipses-of-" + title );
    
    // Define a dat aseries with the current mean "ruler" lines
    std::string mean_ruler_line_ds
      = create_mean_ruler_line_dataseries
      ( pp,
	ptree(),
	"rulers-of-" + title );
    
    // Define a data series with the mixutre perido gamma on
    // the mixture ruler line
    std::string ruler_period_distribution_ds
      = create_ruler_period_distribution_dataseries
      ( pp,
	ptree(),
	"period-gammas-of-" + title );
    
    // Define a data series with the ruler length gamma on the
    // ruler lines
    std::string ruler_length_distribution_ds
      = create_ruler_length_distribution_dataseries
      ( pp,
	ptree(),
	"ruler-length-gammas-of-" + title );
    
    // Define a data series with the ellipses of the ruler start
    // positions
    std::string ruler_start_distribution_ds
      = create_ruler_start_distribution_dataseries
      ( pp,
	ptree(),
	"ruler-start-ellipses-of-" + title );
    
    // Define a data series with the ruler direction distribution
    // std::string ruler_direction_distribution_ds
    //   = create_ruler_direction_distribution_dataseries
    //   ( pp,
    // 	ptree(),
    // 	"ruler-directions-of-" + title );

    // define a data series for the projection of the observations to
    // the actual mixture rulers they come from
    // This iwll dhow the observation->mixture mapping visually
    std::string proj_ds
      = create_observation_projection_dataseries
      ( pp,
	ptree(),
	"observation-projections-of-" + title );

    // create some predefined configs for plots (mostly the styles)
    ptree range;
    range.put( "x.min", pp.window().start.coordinate[0] );
    range.put( "x.max", pp.window().end.coordinate[0] );
    range.put( "y.min", pp.window().start.coordinate[1] );
    range.put( "y.max", pp.window().end.coordinate[1] );
    ptree window_style;
    window_style.put( "plot_prefix", "plot" );
    window_style.put( "gnuplot.style", "lines" );
    window_style.put_child( "range", range );
    window_style.put( "gnuplot.keep_gnuplot_files", true );
    ptree dots_style;
    dots_style.put( "plot_prefix", "plot" );
    dots_style.put( "gnuplot.style", "dots" );
    dots_style.put_child( "range", range );
    dots_style.put( "hidden", true );
    ptree point_style;
    point_style.put( "plot_prefix", "plot" );
    point_style.put( "gnuplot.style", "points" );
    point_style.put_child( "range", range );
    point_style.put( "hidden", true );
    ptree ellipse_style;
    ellipse_style.put( "plot_prefix", "plot" );
    ellipse_style.put( "gnuplot.style", "ellipses" );
    ellipse_style.add( "wanted_attributes.", "x" );
    ellipse_style.add( "wanted_attributes.", "y" );
    ellipse_style.add( "wanted_attributes.", "major_diameter" );
    ellipse_style.add( "wanted_attributes.", "minor_diameter" );
    ellipse_style.add( "wanted_attributes.", "angle" );
    ellipse_style.put_child( "range", range );
    ellipse_style.put( "hidden", true );
    ptree line_style;
    line_style.put( "plot_prefix", "plot" );
    line_style.put( "gnuplot.style", "lines" );
    line_style.put_child( "range", range );
    line_style.put( "hidden", true );
    ptree ruler_style;
    ruler_style.put( "plot_prefix", "plot" );
    ruler_style.put( "gnuplot.style", "vectors" );
    ruler_style.add( "wanted_attributes.", "x" );
    ruler_style.add( "wanted_attributes.", "y" );
    ruler_style.add( "wanted_attributes.", "dx" );
    ruler_style.add( "wanted_attributes.", "dy" );
    ruler_style.put_child( "range", range );
    ruler_style.put( "hidden", true );
    ptree dir_style;
    dir_style.put( "plot_prefix", "plot" );
    dir_style.put( "gnuplot.style", "bars" );
    dir_style.put_child( "range", range );
    dir_style.put( "hidden", true );
    ptree thin_dashed_line_style;
    thin_dashed_line_style.put( "plot_prefix", "plot" );
    thin_dashed_line_style.put( "gnuplot.style", "lines" );
    thin_dashed_line_style.put( "gnuplot.linestyle", "0" );
    thin_dashed_line_style.put( "gnuplot.linewidth", "0.5" );
    thin_dashed_line_style.put_child( "range", range );
    thin_dashed_line_style.put( "hidden", true );

    

    
    
    // Ok, create a plot for each one of these data series,
    // and a compound plot with all of these plots together
    std::vector< std::string > plot_ids;
    plot_ids.push_back( create_plot( dots_style,
     				     { obs_ds } ) );
    plot_ids.push_back( create_plot( point_style,
				     { mean_tick_ds } ) );
    plot_ids.push_back( create_plot( ellipse_style,
				     { gaussian_mixture_ellipse_ds } ) );
    plot_ids.push_back( create_plot( ruler_style,
				     { mean_ruler_line_ds } ) );
    plot_ids.push_back( create_plot( line_style,
				     { ruler_period_distribution_ds } ) );
    plot_ids.push_back( create_plot( line_style,
				     { ruler_length_distribution_ds } ) );
    plot_ids.push_back( create_plot( ellipse_style,
				     { ruler_start_distribution_ds } ) );
    // plot_ids.push_back( create_plot( dir_style,
    // 				     { ruler_direction_distribution_ds } ) );
    plot_ids.push_back( create_plot( thin_dashed_line_style,
				     { proj_ds } ) );
   
    std::string compound_plot
      = create_plot( window_style,
		     { window_ds },
		     title );
    for( std::string pid : plot_ids ) {
      add_plot_to_plot( pid, compound_plot );
    }
    
    return compound_plot;
  }

  //===============================================================
  //===============================================================
  //===============================================================
  //===============================================================
  //===============================================================
  //===============================================================
  //===============================================================
  //===============================================================
  //===============================================================
  //===============================================================
  //===============================================================
  //===============================================================
  //===============================================================
  //===============================================================
  //===============================================================
  //===============================================================
  //===============================================================
  //===============================================================
  //===============================================================
  //===============================================================
  //===============================================================
  //===============================================================


}
