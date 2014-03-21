#if !defined( __P2L_RULER_POINT_PROCESS_flat_parameters_HPP__ )
#define __P2L_RULER_POINT_PROCESS_flat_parameters_HPP__

#include <math-core/geom.hpp>

namespace ruler_point_process {

  
  //====================================================================

  // Description:
  // encodes a point into a flat version of itself
  math_core::nd_point_t encode_point( const math_core::nd_point_t& p );

  //====================================================================

  // description:
  // decodes a point from a falt point
  math_core::nd_point_t decode_point( const math_core::nd_point_t& flat );

  //====================================================================

  // Description:
  // encoides a netaive region into a lat point
  math_core::nd_point_t 
  encode_negative_region( const math_core::nd_aabox_t& r );

  //====================================================================

  // Description:
  // Decode negative region from flat point
  math_core::nd_aabox_t 
  decode_negative_region( const math_core::nd_point_t& flat );

  //====================================================================

}

#endif

