
#include <point-process-experiment-core/experiment_utils.hpp>
#include <ruler-point-process/ruler_point_process.hpp>
#include <planner-core/shortest_path_next_planner.hpp>


using namespace point_process_core;
using namespace ruler_point_process;
using namespace planner_core;
using namespace math_core;
using namespace probability_core;
using namespace point_process_experiment_core;


namespace models {

  //=====================================================================

  boost::shared_ptr<point_process_core::mcmc_point_process_t>
  ruler_1d_small_001( const math_core::nd_aabox_t& window )
  {
    int dim = 1;
    ruler_point_process_model_t model;
    model.prior_ruler_start_mean = zero_point( dim );
    model.prior_ruler_direction_mean = zero_point( dim );
    model.alpha = 1;
    model.precision_distribution.shape = 500000;
    model.precision_distribution.rate = 1000;
    model.period_distribution.p = pow(2.15,4);
    model.period_distribution.q = 2.15*4;
    model.period_distribution.r = 4;
    model.period_distribution.s = 4;
    model.ruler_length_distribution.p = pow(10,4);
    model.ruler_length_distribution.q = 10 * 4;
    model.ruler_length_distribution.r = 4;
    model.ruler_length_distribution.s = 4;
    model.ruler_start_mean_distribution.dimension = dim;
    model.ruler_start_mean_distribution.means.push_back( 0.4 );
    model.ruler_start_mean_distribution.covariance = to_dense_mat( Eigen::MatrixXd::Identity(dim,dim) * (0.4*0.4) );
    model.ruler_start_precision_distribution.shape = 5000;
    model.ruler_start_precision_distribution.rate = 100;
    model.ruler_direction_mean_distribution.dimension = dim;
    model.ruler_direction_mean_distribution.means.push_back( 10 );
    model.ruler_direction_mean_distribution.covariance = to_dense_mat( Eigen::MatrixXd::Identity(dim,dim) * 1 );
    model.ruler_direction_precision_distribution.shape = 5000;
    model.ruler_direction_precision_distribution.rate = 100;
    std::vector<nd_point_t> init_points;
    init_points.push_back( point( 0.0 ) );
    init_points.push_back( point( 1.0 ) );
    boost::shared_ptr<ruler_point_process_t> process = 
      boost::shared_ptr<ruler_point_process_t>
      ( new ruler_point_process_t( window,
				   model,
				   init_points ) );
    boost::shared_ptr<mcmc_point_process_t> planner_process
      = boost::shared_ptr<mcmc_point_process_t>( process );
      
    return planner_process;
  }
    


  //=====================================================================

  boost::shared_ptr<point_process_core::mcmc_point_process_t>
  ruler_2d_small_001( const math_core::nd_aabox_t& window )
  {
    int dim = 2;
    ruler_point_process_model_t model;
    model.prior_ruler_start_mean = zero_point( dim );
    model.prior_ruler_direction_mean = zero_point( dim );
    model.alpha = 1;
    model.precision_distribution.shape = 500000;
    model.precision_distribution.rate = 1000;
    model.period_distribution.p = pow(2.15,40);
    model.period_distribution.q = 2.15*40;
    model.period_distribution.r = 40;
    model.period_distribution.s = 40;
    model.ruler_length_distribution.p = pow(10,40);
    model.ruler_length_distribution.q = 10 * 40;
    model.ruler_length_distribution.r = 40;
    model.ruler_length_distribution.s = 40;
    model.ruler_start_mean_distribution.dimension = dim;
    model.ruler_start_mean_distribution.means.push_back( 0.4 );
    model.ruler_start_mean_distribution.means.push_back( 0.4 );
    model.ruler_start_mean_distribution.covariance = to_dense_mat( Eigen::MatrixXd::Identity(dim,dim) * (0.4*0.4) );
    model.ruler_start_precision_distribution.shape = 5000;
    model.ruler_start_precision_distribution.rate = 100;
    model.ruler_direction_mean_distribution.dimension = dim;
    model.ruler_direction_mean_distribution.means.push_back( 10 );
    model.ruler_direction_mean_distribution.means.push_back( 10 );
    model.ruler_direction_mean_distribution.covariance = to_dense_mat( Eigen::MatrixXd::Identity(dim,dim) * 1 );
    model.ruler_direction_precision_distribution.shape = 5000;
    model.ruler_direction_precision_distribution.rate = 100;
    std::vector<nd_point_t> init_points;
    init_points.push_back( point( 0.0, 0.0 ) );
    init_points.push_back( point( 1.0, 1.0 ) );
    boost::shared_ptr<ruler_point_process_t> process = 
      boost::shared_ptr<ruler_point_process_t>
      ( new ruler_point_process_t( window,
				   model,
				   init_points ) );
    boost::shared_ptr<mcmc_point_process_t> planner_process
      = boost::shared_ptr<mcmc_point_process_t>( process );
      
    return planner_process;
  }

  //=====================================================================

  boost::shared_ptr<point_process_core::mcmc_point_process_t>
  ruler_2d_small_002( const math_core::nd_aabox_t& window )
  {
    int dim = 2;
    ruler_point_process_model_t model;
    model.prior_ruler_start_mean = zero_point( dim );
    model.prior_ruler_direction_mean = zero_point( dim );
    model.alpha = 1;
    model.precision_distribution.shape = 500;
    model.precision_distribution.rate = 10;
    model.period_distribution.p = pow(2.15,1);
    model.period_distribution.q = 2.15*1;
    model.period_distribution.r = 1;
    model.period_distribution.s = 1;
    model.ruler_length_distribution.p = pow(10,1);
    model.ruler_length_distribution.q = 10 * 1;
    model.ruler_length_distribution.r = 1;
    model.ruler_length_distribution.s = 1;
    model.ruler_start_mean_distribution.dimension = dim;
    model.ruler_start_mean_distribution.means.push_back( 0.4 );
    model.ruler_start_mean_distribution.means.push_back( 0.4 );
    model.ruler_start_mean_distribution.covariance = to_dense_mat( Eigen::MatrixXd::Identity(dim,dim) * (0.4*0.4) );
    model.ruler_start_precision_distribution.shape = 500;
    model.ruler_start_precision_distribution.rate = 10;
    model.ruler_direction_mean_distribution.dimension = dim;
    model.ruler_direction_mean_distribution.means.push_back( 10 );
    model.ruler_direction_mean_distribution.means.push_back( 10 );
    model.ruler_direction_mean_distribution.covariance = to_dense_mat( Eigen::MatrixXd::Identity(dim,dim) * 1 );
    model.ruler_direction_precision_distribution.shape = 500;
    model.ruler_direction_precision_distribution.rate = 10;
    std::vector<nd_point_t> init_points;
    init_points.push_back( point( 0.0, 0.0 ) );
    init_points.push_back( point( 1.0, 1.0 ) );
    boost::shared_ptr<ruler_point_process_t> process = 
      boost::shared_ptr<ruler_point_process_t>
      ( new ruler_point_process_t( window,
				   model,
				   init_points ) );
    boost::shared_ptr<mcmc_point_process_t> planner_process
      = boost::shared_ptr<mcmc_point_process_t>( process );
      
    return planner_process;
  }

  //=====================================================================

  boost::shared_ptr<point_process_core::mcmc_point_process_t>
  ruler_2d_small_003( const math_core::nd_aabox_t& window )
  {
    int dim = 2;
    ruler_point_process_model_t model;
    model.prior_ruler_start_mean = zero_point( dim );
    model.prior_ruler_direction_mean = zero_point( dim );
    model.alpha = 1;
    model.precision_distribution.shape = 500;
    model.precision_distribution.rate = 10;
    model.period_distribution.p = pow(2.15,1);
    model.period_distribution.q = 2.15*1;
    model.period_distribution.r = 1;
    model.period_distribution.s = 1;
    model.ruler_length_distribution.p = pow(10,1);
    model.ruler_length_distribution.q = 10 * 1;
    model.ruler_length_distribution.r = 1;
    model.ruler_length_distribution.s = 1;
    model.ruler_start_mean_distribution.dimension = dim;
    model.ruler_start_mean_distribution.means.push_back( 0.4 );
    model.ruler_start_mean_distribution.means.push_back( 0.4 );
    model.ruler_start_mean_distribution.covariance = to_dense_mat( Eigen::MatrixXd::Identity(dim,dim) * (0.4*0.4) );
    model.ruler_start_precision_distribution.shape = 500;
    model.ruler_start_precision_distribution.rate = 10;
    model.ruler_direction_mean_distribution.dimension = dim;
    model.ruler_direction_mean_distribution.means.push_back( 1 );
    model.ruler_direction_mean_distribution.means.push_back( 1 );
    model.ruler_direction_mean_distribution.covariance = to_dense_mat( Eigen::MatrixXd::Identity(dim,dim) * 1 );
    model.ruler_direction_precision_distribution.shape = 500;
    model.ruler_direction_precision_distribution.rate = 10;
    std::vector<nd_point_t> init_points;
    init_points.push_back( point( 0.7, 0.7 ) );
    init_points.push_back( point( 0.2, 0.2 ) );
    init_points.push_back( point( 1.2, 1.2 ) );
    init_points.push_back( point( 3.7, 3.0 ) );
    init_points.push_back( point( 2.2, 2.2 ) );
    init_points.push_back( point( 6.7, 6.7 ) );
    boost::shared_ptr<ruler_point_process_t> process = 
      boost::shared_ptr<ruler_point_process_t>
      ( new ruler_point_process_t( window,
				   model,
				   init_points ) );
    process->set_liklihood_algorithm( mean_likelihood_approximation );
    boost::shared_ptr<mcmc_point_process_t> planner_process
      = boost::shared_ptr<mcmc_point_process_t>( process );
      
    return planner_process;
  }


    
  //=====================================================================

  boost::shared_ptr<point_process_core::mcmc_point_process_t>
  ruler_2d_small_004( const math_core::nd_aabox_t& window )
  {
    int dim = 2;
    ruler_point_process_model_t model;
    model.prior_ruler_start_mean = zero_point( dim );
    model.prior_ruler_direction_mean = zero_point( dim );
    model.alpha = 2;
    model.precision_distribution.shape = 500;
    model.precision_distribution.rate = 10;
    model.period_distribution.p = pow(6.0,1);
    model.period_distribution.q = 6.0*1;
    model.period_distribution.r = 1;
    model.period_distribution.s = 1;
    model.ruler_length_distribution.p = pow(10.0,1);
    model.ruler_length_distribution.q = 10.0 * 1;
    model.ruler_length_distribution.r = 1;
    model.ruler_length_distribution.s = 1;
    model.ruler_start_mean_distribution.dimension = dim;
    model.ruler_start_mean_distribution.means.push_back( 0.4 );
    model.ruler_start_mean_distribution.means.push_back( 0.4 );
    model.ruler_start_mean_distribution.covariance = to_dense_mat( Eigen::MatrixXd::Identity(dim,dim) * (0.4*0.4) );
    model.ruler_start_precision_distribution.shape = 50;
    model.ruler_start_precision_distribution.rate = 1;
    model.ruler_direction_mean_distribution.dimension = dim;
    model.ruler_direction_mean_distribution.means.push_back( 0.0 );
    model.ruler_direction_mean_distribution.means.push_back( 0.0 );
    model.ruler_direction_mean_distribution.covariance = to_dense_mat( Eigen::MatrixXd::Identity(dim,dim) * 1 );
    model.ruler_direction_precision_distribution.shape = 50;
    model.ruler_direction_precision_distribution.rate = 10;
    std::vector<nd_point_t> init_points;
    init_points.push_back( point( 0.7, 0.7 ) );
    init_points.push_back( point( 0.2, 0.2 ) );
    init_points.push_back( point( 1.2, 1.2 ) );
    init_points.push_back( point( 3.7, 3.0 ) );
    init_points.push_back( point( 2.2, 2.2 ) );
    init_points.push_back( point( 6.7, 6.7 ) );
    boost::shared_ptr<ruler_point_process_t> process = 
      boost::shared_ptr<ruler_point_process_t>
      ( new ruler_point_process_t( window,
				   model,
				   init_points ) );
    boost::shared_ptr<mcmc_point_process_t> planner_process
      = boost::shared_ptr<mcmc_point_process_t>( process );
      
    return planner_process;
  }

  //===========================================================================
    

}

//===========================================================================

namespace planners {

  //===========================================================================

  boost::shared_ptr<planner_core::grid_planner_t>
  shortest_path_next_planner_001
  (boost::shared_ptr<point_process_core::mcmc_point_process_t>& model)
  {
    grid_planner_parameters_t planner_params;
    planner_params.burnin_mcmc_iterations = 1;
    planner_params.update_model_mcmc_iterations = 1;
    planner_params.grid_cell_size = 1.0;
    entropy_estimator_parameters_t entropy_params;
    entropy_params.num_samples = 2;
    sampler_planner_parameters_t sampler_planner_params;
    sampler_planner_params.num_samples_of_observations = 1;
    sampler_planner_params.num_samples_of_point_sets = 1;
    sampler_planner_params.num_skip_between_point_set_samples = 0;
    double prob_thresh = 0.6;
    boost::shared_ptr<grid_planner_t> planner
      = boost::shared_ptr<grid_planner_t>
      (
       new shortest_path_next_planner ( model,
					planner_params,
					entropy_params,
					sampler_planner_params,
					prob_thresh)
       );
    
    return planner;
  }
  

  //===========================================================================
  //===========================================================================
  //===========================================================================
  //===========================================================================
  //===========================================================================
  //===========================================================================
  //===========================================================================
  

}


//===========================================================================

namespace worlds {

  //===========================================================================

  std::vector<nd_point_t>
  groundtruth_1d_001()
  {
    std::vector<nd_point_t> points;
    points.push_back( point( 0.25 ) );
    points.push_back( point( 2.25 ) );
    points.push_back( point( 4.25 ) );
    points.push_back( point( 6.25 ) );
    points.push_back( point( 8.25 ) );
    points.push_back( point( 0.55 ) );
    points.push_back( point( 2.55 ) );
    points.push_back( point( 4.55 ) );
    points.push_back( point( 6.55 ) );
    points.push_back( point( 8.55 ) );
    return points;
  }
  
  nd_aabox_t
  window_1d_001()
  {
    return aabox( point( 0.0 ), point( 10.0 ) );
  }
  
  
  //===========================================================================

  std::vector<nd_point_t>
  groundtruth_2d_001()
  {
    std::vector<nd_point_t> points;
    points.push_back( point( 0.25, 0.25 ) );
    points.push_back( point( 2.25, 2.25 ) );
    points.push_back( point( 4.25, 4.25 ) );
    points.push_back( point( 6.25, 6.25 ) );
    points.push_back( point( 8.25, 8.25 ) );
    points.push_back( point( 0.55, 0.55 ) );
    points.push_back( point( 2.55, 2.55 ) );
    points.push_back( point( 4.55, 4.55 ) );
    points.push_back( point( 6.55, 6.55 ) );
    points.push_back( point( 8.55, 8.55 ) );
    return points;
  }

  nd_aabox_t
  window_2d_001()
  {
    return aabox( point( 0.0, 0.0 ), point( 10.0, 10.0 ) );
  }

  //===========================================================================
  //===========================================================================
  //===========================================================================
  //===========================================================================
  //===========================================================================
  //===========================================================================
  //===========================================================================
  //===========================================================================
  //===========================================================================
  //===========================================================================
  //===========================================================================
  //===========================================================================
  //===========================================================================
  //===========================================================================
  //===========================================================================

}

//===========================================================================


void register_experiments()
{

  register_world
    ( "test::world::1d_001",
      worlds::groundtruth_1d_001,
      worlds::window_1d_001 );
  register_world
    ( "test::world::2d_001",
      worlds::groundtruth_2d_001,
      worlds::window_2d_001 );		

  boost::function< boost::shared_ptr<point_process_core::mcmc_point_process_t> (const math_core::nd_aabox_t&)> model_f;
  model_f = models::ruler_1d_small_001;
  register_model
    ( "test::model::ruler_1d_small_001",
      model_f);
  model_f = models::ruler_2d_small_001;
  register_model
    ( "test::model::ruler_2d_small_001",
      model_f);
  model_f = models::ruler_2d_small_002;
  register_model
    ( "test::model::ruler_2d_small_002",
      model_f);
  model_f = models::ruler_2d_small_003;
  register_model
    ( "test::model::ruler_2d_small_003",
      model_f);
  model_f = models::ruler_2d_small_004;
  register_model
    ( "test::model::ruler_2d_small_004",
      model_f);



  boost::function< boost::shared_ptr<planner_core::grid_planner_t> ( boost::shared_ptr<point_process_core::mcmc_point_process_t>&) > planner_f;
  planner_f = planners::shortest_path_next_planner_001;
  register_planner
    ( "test::planner::shortest_path_next_planner_001",
      planner_f);
}


//===========================================================================
