cmake_minimum_required(VERSION 2.6.0)

# pull in the pods macros. See cmake/pods.cmake for documentation
set(POD_NAME ruler-point-process)
include(cmake/pods.cmake)

# automatically build LCM types.  This also defines a number of CMake
# variables, see cmake/lcmtypes.cmake for details
include(cmake/lcmtypes.cmake)
lcmtypes_build()

# build two LCM test programs
add_executable(send_message src/send_message.c)
pods_use_pkg_config_packages(send_message lcm lcmtypes_${POD_NAME})

add_executable(listener src/listener.c)
pods_use_pkg_config_packages(listener lcm lcmtypes_${POD_NAME})

# make exeutables public
pods_install_executables(send_message listener)



add_definitions( -std=c++0x -pedantic -Wall )
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3")



add_library( ruler-point-process SHARED
  src/model.cpp
  src/mcmc.cpp
)
pods_install_headers(
  src/model.hpp
  src/mcmc.hpp
  src/ruler_point_process.hpp
  DESTINATION
  ruler-point-process)
pods_install_libraries( ruler-point-process )
pods_use_pkg_config_packages(ruler-point-process probability-core gsl math-core lcmtypes_probability-core lcmtypes_math-core boost-1.51.0)
pods_install_pkg_config_file(ruler-point-process
    CFLAGS
    LIBS -lruler-point-process
    REQUIRES math-core probability-core point-process-core boost-1.51.0
    VERSION 0.0.1)


add_subdirectory( test )