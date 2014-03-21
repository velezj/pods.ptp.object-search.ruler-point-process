cmake_minimum_required(VERSION 2.6.0)

# pull in the pods macros. See cmake/pods.cmake for documentation
set(POD_NAME p2l-ruler-point-process)
include(cmake/pods.cmake)

# automatically build LCM types.  This also defines a number of CMake
# variables, see cmake/lcmtypes.cmake for details
include(cmake/lcmtypes.cmake)
lcmtypes_build()


add_definitions( -std=c++0x -Wall -fdiagnostics-show-option -Wno-unused-local-typedefs -g )
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3")

option ( USE_PEDANTIC "Turn on -pendantic mode in gcc. This will spit out *lots* of warnings from lcm :-(, but hopefully none from the rest of the code" OFF)
if( USE_PEDANTIC )
    add_definitions( -pedantic )
endif (USE_PEDANTIC)



add_library( p2l-ruler-point-process SHARED
  src/model.cpp
  src/mcmc.cpp
  src/plots.cpp
  src/ruler_point_process.cpp
  src/gem_k_ruler_process.cpp
  src/gem_k_strip_process.cpp
)
pods_install_headers(
  src/model.hpp
  src/mcmc.hpp
  src/ruler_point_process.hpp
  src/plots.hpp
  src/gem_k_ruler_process.hpp
  src/gem_k_strip_process.hpp
  DESTINATION
  ruler-point-process)
pods_install_libraries( p2l-ruler-point-process )
pods_use_pkg_config_packages(p2l-ruler-point-process 
    p2l-common
    p2l-probability-core 
    gsl-1.16 
    p2l-math-core 
    lcmtypes_p2l-probability-core 
    lcmtypes_p2l-math-core 
    boost-1.54.0
    boost-1.54.0-filesystem
    boost-1.54.0-exception
    plot-server-api )
pods_install_pkg_config_file(p2l-ruler-point-process
    CFLAGS
    LIBS -lp2l-ruler-point-process
    REQUIRES p2l-common p2l-math-core p2l-probability-core p2l-point-process-core boost-1.54.0 boost-1.54.0-filesystem boost-1.54.0-exception plot-server-api
    VERSION 0.0.1)


add_subdirectory( test )
