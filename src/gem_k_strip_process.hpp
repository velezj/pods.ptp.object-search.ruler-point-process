
#if !defined( __P2L_RULER_PROCESS_gem_k_strip_process_HPP__ )
#define __P2L_RULER_PROCESS_gem_k_strip_process_HPP__

#include <point-process-core/point_process.hpp>
#include <probability-core/EM.hpp>
#include <math-core/polynomial.hpp>
#include <iosfwd>


namespace ruler_point_process {

  using namespace point_process_core;
  using namespace probability_core;
  using namespace math_core;

  //====================================================================

  // Description:
  // A parameter setting for a single "strip"
  struct strip_t
  {
    math_core::nd_point_t start;
    math_core::polynomial_t manifold;
    size_t num_ticks;
    double length_scale;
    double spread;
  };

  std::ostream& operator<< (std::ostream& os, const strip_t& r );

  //====================================================================

  // Description:
  // The parameters for a gem_k_ruler_process_t
  struct gem_k_strip_process_parmaeters_t
  {
    GEM_parameters_t gem;
    size_t num_strips;
    size_t strip_poly_order;
    size_t num_gem_restarts;
  };

  //====================================================================

  // Description:
  // A ruler process which has an explicit number of strips (k)
  // of a particular dimension (D)
  // and uses Generalized EM (GEM) to fit a cluster model of
  // the strips to the data.
  class gem_k_strip_process_t : public point_process_t<gem_k_strip_process_t>
  {
  public:

    // Description:
    // Create a new gem_k_strip_process_t
    gem_k_strip_process_t( const math_core::nd_aabox_t& window,
			   gem_k_strip_process_parmaeters_t& params,
			   const std::vector<math_core::nd_point_t>& obs,
			   const std::vector<math_core::nd_aabox_t>& neg_obs )
      : _window( window ),
	_params( params ),
	_ndim( window.start.n ),
	_negative_observations( neg_obs )
    {
      this->add_observations( obs );
    }

    gem_k_strip_process_t( const gem_k_strip_process_t&g )
      : _window(g._window),
	_params(g._params),
	_ndim(g._ndim),
	_observations(g._observations),
	_strips(g._strips),
	_negative_observations(g._negative_observations)
    {}

    boost::shared_ptr<gem_k_strip_process_t>
    clone() const
    {
      return boost::shared_ptr<gem_k_strip_process_t>
	(new gem_k_strip_process_t(*this));
    }

    virtual ~gem_k_strip_process_t() {}


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
    
    std::vector<strip_t> strips() const
    { return _strips; }

    std::vector<double> mixture_weights() const
    { return _mixture_weights; }

    // Description:
    // Return the lilelihod of the current rulers/mixture-weights
    // given the data (observations and negative observations)
    double likelihood() const;

    // Description:
    // for testing only!
    void _set_strips( const std::vector<strip_t>& r )
    { _strips = r; }
    void _set_mixture_weights( const std::vector<double>& w )
    { _mixture_weights = w; }
    
  protected:

    // Description:
    // Converts a strip to a flat vector of parameters and back
    static std::vector<double> to_flat_vector( const strip_t& r );
    static strip_t to_strip( const std::vector<double>& v,
			     const size_t point_dim,
			     const size_t poly_dim);

    std::vector<nd_point_t> 
    ticks_for_strip( const strip_t& r ) const;

    // Description:
    // HE workhorse: run GEM to fit rulers to data
    void _run_single_GEM();
    void _run_GEM();

    // Description:
    // the likelihood function given a strip for a data point.
    // Below is same version just using std::vector<double> instaed of ruler
    double lik_single_point_single_strip
    ( const math_core::nd_point_t& single_x,
      const strip_t& strip ) const;

    double lik_single_point_single_strip_flat
    ( const math_core::nd_point_t& single_x,
      const std::vector<double>& params ) const
    {
      return lik_single_point_single_strip(single_x,
					   to_strip(params, _ndim,
						    _params.strip_poly_order));
    }

    // Description:
    // The likelihood function for a negative reagion given
    // a ruler
    double lik_negative_region_strip
    ( const math_core::nd_aabox_t& region,
      const strip_t& strip ) const;
    double lik_negative_region_strip_flat
    ( const math_core::nd_aabox_t& region,
      const std::vector<double>& params ) const
    {
      return lik_negative_region_strip( region,
					to_strip(params,_ndim,
						 _params.strip_poly_order));
    }

    // Description:
    // A "mixed" data inpiut lilelihood function which essentially
    // fowards to one above
    double lik_mixed_strip_flat
    ( const math_core::nd_point_t& flat,
      const std::vector<double>& params ) const;


    // Description:
    // create the bounds for the flat parmeters
    void create_bounds
    ( const size_t& num_strips,
      std::vector<std::vector<double> >& lb,
      std::vector<std::vector<double> >& ub ) const;

    std::vector<strip_t>
    create_initial_strips() const;
    
    
    math_core::nd_aabox_t _window;
    std::vector<math_core::nd_point_t> _observations;
    gem_k_strip_process_parmaeters_t _params;
    std::vector<strip_t> _strips;
    size_t _ndim; // dimension of nd_point_t
    std::vector<math_core::nd_aabox_t> _negative_observations;
    std::vector<double> _mixture_weights;
    
  };

  //====================================================================

  //====================================================================


}

#endif
