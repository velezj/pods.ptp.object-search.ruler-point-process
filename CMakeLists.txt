cmake_minimum_required(VERSION 2.6.0)

# pull in the pods macros. See cmake/pods.cmake for documentation
set(POD_NAME object-search.ruler-point-process)
include(cmake/pods.cmake)



add_definitions( -std=c++0x -Wall -fdiagnostics-show-option -Wno-unused-local-typedefs -fPIC )
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3")

option ( USE_PEDANTIC "Turn on -pendantic mode in gcc. This will spit out *lots* of warnings from lcm :-(, but hopefully none from the rest of the code" OFF)
if( USE_PEDANTIC )
    add_definitions( -pedantic )
endif (USE_PEDANTIC)



add_library( object-search.ruler-point-process SHARED
  src/model.cpp
  src/mcmc.cpp
  src/plots.cpp
  src/flat_parameters.cpp
  src/ruler_point_process.cpp
  src/gem_k_ruler_process.cpp
  src/gem_k_strip_process.cpp
)
pods_install_headers(
  src/model.hpp
  src/mcmc.hpp
  src/ruler_point_process.hpp
  src/plots.hpp
  src/flat_parameters.hpp
  src/gem_k_ruler_process.hpp
  src/gem_k_strip_process.hpp
  src/tick_models.hpp
  src/gem_k_tick_process.hpp
  src/gem_k_tick_process.cpp
  DESTINATION
  ruler-point-process)
pods_install_libraries( object-search.ruler-point-process )
pods_use_pkg_config_packages(object-search.ruler-point-process 
    object-search.common
    object-search.probability-core 
    gsl-1.16 
    object-search.math-core 
    boost-1.54.0
    boost-1.54.0-filesystem
    boost-1.54.0-exception
    plot-server-api )
pods_install_pkg_config_file(object-search.ruler-point-process
    CFLAGS
    LIBS -lobject-search.ruler-point-process
    REQUIRES object-search.common object-search.math-core object-search.probability-core object-search.point-process-core boost-1.54.0 boost-1.54.0-filesystem boost-1.54.0-exception plot-server-api
    VERSION 0.0.1)


add_subdirectory( test )
