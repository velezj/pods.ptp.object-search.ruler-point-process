
#include "ruler_point_process.hpp"
#include "plots.hpp"
#include <iostream>


namespace ruler_point_process {

    std::string
    ruler_point_process_t::plot( const std::string& title ) const
    {
      std::string pid = plot_ruler_point_process( *this, title );
      std::cout << "Ruler-Point-Process Plot: " << pid << std::endl;
      return pid;
    }


}
