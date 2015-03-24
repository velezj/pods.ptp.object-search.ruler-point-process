
#include "flat_parameters.hpp"
#include <limits>
#include <cassert>


namespace ruler_point_process {

    //====================================================================

  // Description:
  // encodes a point into a flat version of itself
  math_core::nd_point_t encode_point( const math_core::nd_point_t& p )
  {
    math_core::nd_point_t res;
    res.n = 2 * p.n + 1;
    res.coordinate.push_back( 1.0 );
    res.coordinate.insert( res.coordinate.end(),
			   p.coordinate.begin(),
			   p.coordinate.end() );
    for( size_t i = 0; i < p.n; ++i ) {
      res.coordinate.push_back( std::numeric_limits<double>::quiet_NaN() );
    }
    return res;
  }

  // description:
  // decodes a point from a falt point
  math_core::nd_point_t decode_point( const math_core::nd_point_t& flat )
  {
    assert( flat.coordinate[0] > 0 );
    assert( (flat.n-1) % 2 == 0 );
    size_t n = (flat.n-1)/2;
    math_core::nd_point_t p;
    p.n = n;
    p.coordinate.insert( p.coordinate.end(),
			 flat.coordinate.begin() + 1,
			 flat.coordinate.begin() + 1 + n );
    return p;
  }

  // Description:
  // encoides a netaive region into a lat point
  math_core::nd_point_t encode_negative_region( const math_core::nd_aabox_t& r )
  {
    math_core::nd_point_t res;
    res.n = 2 * r.start.n + 1;
    res.coordinate.push_back( -1.0 );
    res.coordinate.insert( res.coordinate.end(),
			   r.start.coordinate.begin(),
			   r.start.coordinate.end() );
    res.coordinate.insert( res.coordinate.end(),
			   r.end.coordinate.begin(),
			   r.end.coordinate.end() );
    return res;
  }

  // Description:
  // Decode negative region from flat point
  math_core::nd_aabox_t decode_negative_region( const math_core::nd_point_t& flat )
  {
    assert( flat.coordinate[0] < 0 );
    assert( (flat.n-1) % 2 == 0 );
    size_t n = (flat.n-1)/2;
    math_core::nd_point_t start, end;
    start.n = n;
    start.coordinate.insert( start.coordinate.end(),
			     flat.coordinate.begin() + 1,
			     flat.coordinate.begin() + 1 + n );
    end.n = n;
    end.coordinate.insert( end.coordinate.end(),
			   flat.coordinate.begin() + 1 + n,
			   flat.coordinate.end() );
    return aabox( start, end );
  }


  
  // //====================================================================

  // // Description:
  // // encodes a point into a flat version of itself
  // math_core::nd_point_t encode_point( const math_core::nd_point_t& p )
  // {
  //   math_core::nd_point_t res;
  //   res.n = p.n + 1;
  //   res.coordinate.push_back( 1.0 );
  //   res.coordinate.insert( res.coordinate.end(),
  // 			   p.coordinate.begin(),
  // 			   p.coordinate.end() );
  //   return res;
  // }

  // //====================================================================

  // // description:
  // // decodes a point from a falt point
  // math_core::nd_point_t decode_point( const math_core::nd_point_t& flat )
  // {
  //   assert( flat.coordinate[0] > 0 );
  //   size_t n = flat.n - 1;
  //   math_core::nd_point_t p;
  //   p.n = n;
  //   p.coordinate.insert( p.coordinate.end(),
  // 			 flat.coordinate.begin() + 1,
  // 			 flat.coordinate.end() );
  //   return p;
  // }

  // //====================================================================
  
  // // Description:
  // // encoides a netaive region into a lat point
  // math_core::nd_point_t encode_negative_region( const math_core::nd_aabox_t& r )
  // {
  //   math_core::nd_point_t res;
  //   res.n = 2 * r.start.n + 1;
  //   res.coordinate.push_back( -1.0 );
  //   res.coordinate.insert( res.coordinate.end(),
  // 			   r.start.coordinate.begin(),
  // 			   r.start.coordinate.end() );
  //   res.coordinate.insert( res.coordinate.end(),
  // 			   r.end.coordinate.begin(),
  // 			   r.end.coordinate.end() );
  //   return res;
  // }

  // //====================================================================

  // // Description:
  // // Decode negative region from flat point
  // math_core::nd_aabox_t decode_negative_region( const math_core::nd_point_t& flat )
  // {
  //   assert( flat.coordinate[0] < 0 );
  //   assert( (flat.n-1) % 2 == 0 );
  //   size_t n = (flat.n-1)/2;
  //   math_core::nd_point_t start, end;
  //   start.n = n;
  //   start.coordinate.insert( start.coordinate.end(),
  // 			     flat.coordinate.begin() + 1,
  // 			     flat.coordinate.begin() + 1 + n );
  //   end.n = n;
  //   end.coordinate.insert( end.coordinate.end(),
  // 			   flat.coordinate.begin() + 1 + n,
  // 			   flat.coordinate.end() );
  //   return aabox( start, end );
  // }

  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================


}
