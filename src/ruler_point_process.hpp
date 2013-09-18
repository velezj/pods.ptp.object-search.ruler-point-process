#if !defined( __RULER_POINT_PROCESS_RULER_POINT_PROCESS_HPP__ )
#define __RULER_POINT_PROCESS_RULER_POINT_PROCESS_HPP__


#include <point-process-core/point_process.hpp>
#include "model.hpp"
#include "mcmc.hpp"
#include <probability-core/distribution_utils.hpp>
#include <math-core/matrix.hpp>


namespace ruler_point_process {

  // Description:
  // The mcmc_point_process_t subclass for the ruler_point_process
  class ruler_point_process_t : public point_process_core::mcmc_point_process_t
  {

  public: // CREATION

    // Description:
    // Create a new process
    ruler_point_process_t( const math_core::nd_aabox_t& window,
			   const ruler_point_process_model_t& model,
			   const std::vector<math_core::nd_point_t>& obs )
    {
      _state.window = window;
      _state.model = model;
      _state.trace_mcmc = false;
      _state.trace_samples = false;
      _state.iteration = 0;
      this->add_observations( obs );
    }


    // Description:
    // Clones this process
    virtual
    boost::shared_ptr<mcmc_point_process_t>
    clone() const
    {
      return boost::shared_ptr<mcmc_point_process_t>( new ruler_point_process_t( _state ) );
    }

    virtual ~ruler_point_process_t() {}

  public: // API

    // Description:
    // Retrusn the window for this point process
    virtual
    math_core::nd_aabox_t window() const
    {
      return _state.window;
    }


    // Description:
    virtual
    std::vector<math_core::nd_point_t>
    observations() const
    {
      return _state.observations;
    }

    // Description:
    // Sample ffrom the process
    virtual
    std::vector<math_core::nd_point_t>
    sample() const
    {
      return sample_from( _state );
    }
    
    // Description:
    // Run a single mcmc step
    virtual
    void single_mcmc_step()
    {
      mcmc_single_step( _state );
    }
    
    // Description:
    // Add observations to this process
    virtual
    void add_observations( const std::vector<math_core::nd_point_t>& obs )
    {
      int new_obs_start_index = _state.observations.size();
      _state.observations.insert( _state.observations.end(),
				  obs.begin(),
				  obs.end() );

      // the dimensionality of points
      int dim = obs[0].n;

      // Ok, we will make all these observation into a single new 
      // mixture with components sample from the priors
      size_t mixture_index = _state.mixture_gaussians.size();

      // sample new mixture from prior
      probability_core::gaussian_distribution_t zero_gaussian;
      zero_gaussian.dimension = dim;
      for( int i = 0; i < dim; ++i ) {
	zero_gaussian.means.push_back( 1e-5 );
      }
      zero_gaussian.covariance = math_core::to_dense_mat( Eigen::MatrixXd::Identity(dim,dim) * 1e-10 );
      _state.mixture_gaussians.push_back( sample_gaussian_from( zero_gaussian,
							       _state.model.precision_distribution ) );
      _state.mixture_period_gammas.push_back( sample_from( _state.model.period_distribution ) );
      _state.mixture_ruler_length_gammas.push_back( sample_from( _state.model.ruler_length_distribution ) );
      _state.mixture_ruler_start_gaussians.push_back( sample_gaussian_from( _state.model.ruler_start_mean_distribution,
									   _state.model.ruler_start_precision_distribution ));
      _state.mixture_ruler_direction_gaussians.push_back( sample_gaussian_from( _state.model.ruler_direction_mean_distribution,
									       _state.model.ruler_direction_precision_distribution ));
      
      // map all new observations to this new mixture
      std::vector<size_t> obs_indices;
      for( size_t i = 0; i < obs.size(); ++i ) {
	_state.observation_to_mixture.push_back( mixture_index );
	obs_indices.push_back( new_obs_start_index + i );
      }
      _state.mixture_to_observation_indices.push_back( obs_indices );
    }


    // Descripiton:
    // Add a negative observation
    virtual
    void add_negative_observation( const math_core::nd_aabox_t& region )
    {
      _state.negative_observations.push_back( region );
    }

    // Description:
    // Turns on or off mcmc  tracing
    virtual
    void trace_mcmc( const std::string& trace_dir )
    {
      _state.trace_mcmc = true;
      _state.trace_mcmc_dir = trace_dir;
    }
    
    virtual 
    void trace_mcmc_off()
    {
      _state.trace_mcmc = false;
    }


  public: // STATE

    
    // Description:
    // The ruler-point-process-state for this process
    ruler_point_process_state_t _state;

  protected: // CREATION
    
    // Description:
    // Creates a new point process with copy of the given state
    ruler_point_process_t( const ruler_point_process_state_t& state )
      : _state( state )
    {}

  };
  


}

#endif

