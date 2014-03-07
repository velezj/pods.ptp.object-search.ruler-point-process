
#if !defined( __P2L_RULER_PROCESS_gem_k_ruler_process_HPP__ )
#define __P2L_RULER_PROCESS_gem_k_ruler_process_HPP__

#include <point-process-core/point_process.hpp>
#include <probability-core/EM.hpp>
#include <probability-core/negative_binomial.hpp>
#include <iosfwd>


namespace ruler_point_process {

  using namespace point_process_core;
  using namespace probability_core;
  using namespace math_core;

  //====================================================================

  // Description:
  // A "ruler" parameter setting for a single ruler
  struct ruler_t
  {
    math_core::nd_point_t start;
    math_core::nd_direction_t dir;
    size_t num_ticks;
    double length_scale;
    double spread;
  };

  std::ostream& operator<< (std::ostream& os, const ruler_t& r );

  //====================================================================

  // Description:
  // The parameters for a gem_k_ruler_process_t
  struct gem_k_ruler_process_parmaeters_t
  {
    GEM_parameters_t gem;
    size_t num_rulers;
    size_t num_gem_restarts;
  };

  //====================================================================

  // Description:
  // A ruler process which has an explicit number of rulers (k)
  // and uses Generalized EM (GEM) to fit a cluster model of
  // the rulers to the data.
  class gem_k_ruler_process_t : public point_process_t<gem_k_ruler_process_t>
  {
  public:

    // Description:
    // Create a new gem_k_ruler_process_t
    gem_k_ruler_process_t( const math_core::nd_aabox_t& window,
			   gem_k_ruler_process_parmaeters_t& params,
			   const std::vector<math_core::nd_point_t>& obs,
			   const std::vector<math_core::nd_aabox_t>& neg_obs )
      : _window( window ),
	_params( params ),
	_ndim( window.start.n ),
	_negative_observations( neg_obs )
    {
      this->add_observations( obs );
    }

    gem_k_ruler_process_t( const gem_k_ruler_process_t&g )
      : _window(g._window),
	_params(g._params),
	_ndim(g._ndim),
	_observations(g._observations),
	_rulers(g._rulers),
	_negative_observations(g._negative_observations)
    {}

    boost::shared_ptr<gem_k_ruler_process_t>
    clone() const
    {
      return boost::shared_ptr<gem_k_ruler_process_t>
	(new gem_k_ruler_process_t(*this));
    }

    virtual ~gem_k_ruler_process_t() {}


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

  public: // gem_k_ruler_process_t specific API
    
    std::vector<ruler_t> rulers() const
    { return _rulers; }

    std::vector<double> mixture_weights() const
    { return _ruler_mixture_weights; }

    // Description:
    // Return the lilelihod of the current rulers/mixture-weights
    // given the data (observations and negative observations)
    double likelihood() const;

    // Description:
    // for testing only!
    void _set_rulers( const std::vector<ruler_t>& r )
    { _rulers = r; }
    void _set_ruler_mixture_weights( const std::vector<double>& w )
    { _ruler_mixture_weights = w; }
    
  protected:

    // Description:
    // Converts a ruler to a flat vector of parameters and back
    static std::vector<double> to_flat_vector( const ruler_t& r );
    static ruler_t to_ruler( const std::vector<double>& v,
			     const size_t ndim );

    std::vector<nd_point_t> 
    ticks_for_ruler( const ruler_t& r ) const;

    // Description:
    // HE workhorse: run GEM to fit rulers to data
    void _run_single_GEM();
    void _run_GEM();

    // Description:
    // the likelihood function given a ruler for a data point.
    // Below is same version just using std::vector<double> instaed of ruler
    double lik_single_point_single_ruler
    ( const math_core::nd_point_t& single_x,
      const ruler_t& ruler ) const;

    double lik_single_point_single_ruler_flat
    ( const math_core::nd_point_t& single_x,
      const std::vector<double>& ruler_params ) const
    {
      return lik_single_point_single_ruler(single_x,
					   to_ruler(ruler_params, _ndim ));
    }

    // Description:
    // The likelihood function for a negative reagion given
    // a ruler
    double lik_negative_region_ruler
    ( const math_core::nd_aabox_t& region,
      const ruler_t& ruler ) const;
    double lik_negative_region_ruler_flat
    ( const math_core::nd_aabox_t& region,
      const std::vector<double>& ruler_params ) const
    {
      return lik_negative_region_ruler( region,
					to_ruler(ruler_params,_ndim));
    }

    // Description:
    // A "mixed" data inpiut lilelihood function which essentially
    // fowards to one above
    double lik_mixed_ruler_flat
    ( const math_core::nd_point_t& flat,
      const std::vector<double>& ruler_params ) const;


    std::vector<ruler_t>
    create_initial_rulers() const;
    
    
    math_core::nd_aabox_t _window;
    std::vector<math_core::nd_point_t> _observations;
    gem_k_ruler_process_parmaeters_t _params;
    std::vector<ruler_t> _rulers;
    size_t _ndim; // dimension of nd_point_t
    std::vector<math_core::nd_aabox_t> _negative_observations;
    std::vector<double> _ruler_mixture_weights;
    
  };

  //====================================================================

  //====================================================================


}

#endif
