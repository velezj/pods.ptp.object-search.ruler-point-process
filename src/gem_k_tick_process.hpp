
#if !defined( __P2L_RULER_POINT_PROCESS_gem_k_tick_process_HPP__ )
#define __P2L_RULER_POINT_PROCESS_gem_k_tick_process_HPP__

#include <point-process-core/point_process.hpp>
#include <probability-core/EM.hpp>
#include <iosfwd>


namespace ruler_point_process {

  using namespace point_process_core;
  using namespace probability_core;
  using namespace math_core;

  //====================================================================

  // Description:
  // The parameters for a gem_k_tick_process
  template< class TickModelT >
  struct gem_k_tick_process_parmaeters_t
  {
    GEM_parameters_t gem;
    size_t k;
    size_t num_gem_restarts;
    size_t point_dimension;
    double tick_spread;
    typename global_parameters_of<TickModelT>::type  tick_model_parameters;
    std::vector<std::vector<double> > tick_model_flat_lower_bounds;
    std::vector<std::vector<double> > tick_model_flat_upper_bounds;
  };

  //====================================================================

  // Description:
  // A ruler process which has an explicit number of strips (k)
  // of a particular dimension (D)
  // and uses Generalized EM (GEM) to fit a cluster model of
  // the strips to the data.
  template< class TickModelT>
  class gem_k_tick_process_t : 
    public point_process_t<gem_k_tick_process_t<TickModelT> >
  {
  public:

    // Description:
    // Create a new process
    gem_k_tick_process_t( const math_core::nd_aabox_t& window,
			  gem_k_tick_process_parmaeters_t& params,
			  const std::vector<math_core::nd_point_t>& obs,
			  const std::vector<math_core::nd_aabox_t>& neg_obs )
      : _window( window ),
	_params( params ),
	_ndim( window.start.n ),
	_negative_observations( neg_obs )
    {
      this->add_observations( obs );
    }

    gem_k_tick_process_t( const gem_k_tick_process_t&g )
      : _window(g._window),
	_params(g._params),
	_observations(g._observations),
	_negative_observations(g._negative_observations),
	_tick_models(g._tick_models)
    {}

    boost::shared_ptr<gem_k_tick_process_t<TickModelT> >
    clone() const
    {
      return boost::shared_ptr<gem_k_tick_process_t<TickModelT> >
	(new gem_k_tick_process_t<TickModelT>(*this));
    }

    virtual ~gem_k_tick_process_t() {}


  public: // API

    // Description:
    // Retrusn the window for this point process
    virtual
    math_core::nd_aabox_t window() const
    {
      return _window;
    }


    // Description:
    // returns all observations
    virtual
    std::vector<math_core::nd_point_t>
    observations() const
    {
      return _observations;
    }

    // Description:
    // Add observations to this process
    virtual
    void add_observations( const std::vector<math_core::nd_point_t>& obs )
    {
      _observations.insert( _observations.end(),
			    obs.begin(),
			    obs.end() );
      _run_GEM();
    }

    // Descripiton:
    // Add a negative observation
    virtual
    void add_negative_observation( const math_core::nd_aabox_t& region )
    {
      _negative_observations.push_back( region );
    }

    std::vector<math_core::nd_point_t> sample() const;

    virtual
    void print_shallow_trace( std::ostream& out ) const
    {
    }

    virtual
    std::string
    plot( const std::string& title ) const
    { return ""; }

  public: // gem_k_tick_process_t specific API
    
    std::vector<TickModelT> tick_models() const
    { return _tick_models; }

    std::vector<double> mixture_weights() const
    { return _mixture_weights; }

    // Description:
    // Return the lilelihod of the current rulers/mixture-weights
    // given the data (observations and negative observations)
    double likelihood() const;

    // Description:
    // for testing only!
    void _set_tick_models( const std::vector<TickModelT>& r )
    { _tick_models = r; }
    void _set_mixture_weights( const std::vector<double>& w )
    { _mixture_weights = w; }

    
  protected:

    // Description:
    // THE workhorse: run GEM to fit rulers to data
    void _run_single_GEM();
    void _run_GEM();

    // Description:
    // the likelihood function given a strip for a data point.
    // Below is same version just using std::vector<double> instaed of ruler
    double lik_single_point_single_model
    ( const math_core::nd_point_t& single_x,
      const TickModelT& strip ) const;

    double lik_single_point_single_model_flat
    ( const math_core::nd_point_t& single_x,
      const std::vector<double>& params ) const;

    // Description:
    // The likelihood function for a negative reagion given
    // a ruler
    double lik_negative_region
    ( const math_core::nd_aabox_t& region,
      const TickModelT& strip ) const;

    double lik_negative_region_flat
    ( const math_core::nd_aabox_t& region,
      const std::vector<double>& params ) const;

    // Description:
    // A "mixed" data inpiut lilelihood function which essentially
    // fowards to one above
    double lik_mixed_flat
    ( const math_core::nd_point_t& flat,
      const std::vector<double>& params ) const;


    std::vector<TickModelT>
    create_initial_tick_models(const double& margin = 0.1) const;
    
    
    math_core::nd_aabox_t _window;
    gem_k_strip_process_parmaeters_t _params;
    std::vector<math_core::nd_point_t> _observations;
    std::vector<math_core::nd_aabox_t> _negative_observations;
    std::vector<TickModelT> _tick_models;
    std::vector<double> _mixture_weights;
    
  };

  //====================================================================

  //====================================================================
  
  // Description:
  // API for tick_model classes

  template<class TickModelT>
  struct global_parameters_of {
    typedef void type;
  };
  
  
  template<class TickModelT>
  void
  flat_to_model( const std::vector<double>& flat,
		 TickModelT& model,
		 const typename global_parameters_of<TickModelT>::type& params);

  template<class TickModelT>
  std::vector<double>
  model_to_flat( const TickModelT& model,
		 const typename global_parameters_of<TickModelT>::type& params);

  template<class TickModelT>
  std::vector<math_core::nd_point_t>
  ticks_for_model( const TickModelT& model,
		   const typename global_parameters_of<TickModelT>::type& params );

  // Also needed:
  // probability_core::sample_from( uniform_distribution_t<TickModelT> )



  //====================================================================


}

#endif

