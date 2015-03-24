
# add_executable( object-search.ruler-point-process-test-likelihood-montecarlo
#   test-likelihood-montecarlo.cpp )
# pods_use_pkg_config_packages( object-search.ruler-point-process-test-likelihood-montecarlo
#     object-search.math-core 
#     object-search.point-process-core 
#     object-search.probability-core 
#     object-search.ruler-point-process 
#     lcmtypes_object-search.math-core 
#     lcmtypes_object-search.probability-core 
#     lcmtypes_object-search.igmm-point-process )
# pods_install_executables( object-search.ruler-point-process-test-likelihood-montecarlo )


# add_executable( object-search.ruler-point-process-test-mcmc
#   test-mcmc.cpp
# )
# pods_use_pkg_config_packages( object-search.ruler-point-process-test-mcmc
#   object-search.math-core lcmtypes_object-search.math-core
#   object-search.probability-core lcmtypes_object-search.probability-core
#   object-search.point-process-core
#   object-search.point-process-experiment-core
#   object-search.ruler-point-process
# )
# pods_install_executables( object-search.ruler-point-process-test-mcmc )

  


add_executable( object-search.ruler-point-process-test-rp
  test-rp.cpp
)
pods_use_pkg_config_packages( object-search.ruler-point-process-test-rp
  boost-1.54.0
  object-search.math-core
  object-search.probability-core
  object-search.point-process-core
  object-search.point-process-experiment-core
  object-search.ruler-point-process
)
pods_install_executables( object-search.ruler-point-process-test-rp )



add_executable( object-search.ruler-point-process-test-plots
  test-plots.cpp
)
pods_use_pkg_config_packages( object-search.ruler-point-process-test-plots
  boost-1.54.0
  object-search.math-core
  object-search.probability-core
  object-search.point-process-core
  object-search.point-process-experiment-core
  object-search.ruler-point-process
)
pods_install_executables( object-search.ruler-point-process-test-plots )

  
# add_executable( object-search.ruler-point-process-test-gem-k-ruler-process
#   test-gem-k-ruler-process.cpp
# )
# pods_use_pkg_config_packages( object-search.ruler-point-process-test-gem-k-ruler-process
#   boost-1.54.0
#   object-search.math-core lcmtypes_object-search.math-core
#   object-search.probability-core lcmtypes_object-search.probability-core
#   object-search.point-process-core
#   object-search.point-process-experiment-core
#   object-search.ruler-point-process
# )
# pods_install_executables( object-search.ruler-point-process-test-gem-k-ruler-process )


# add_executable( object-search.ruler-point-process-test-gem-k-strip-process
#   test-gem-k-strip-process.cpp
# )
# pods_use_pkg_config_packages( object-search.ruler-point-process-test-gem-k-strip-process
#   boost-1.54.0
#   object-search.math-core lcmtypes_object-search.math-core
#   object-search.probability-core lcmtypes_object-search.probability-core
#   object-search.point-process-core
#   object-search.point-process-experiment-core
#   object-search.ruler-point-process
# )
# pods_install_executables( object-search.ruler-point-process-test-gem-k-strip-process )

  
add_executable( object-search.ruler-point-process-test-gem-k-tick-process
  test-gem-k-tick-process.cpp )
pods_use_pkg_config_packages( object-search.ruler-point-process-test-gem-k-tick-process
  object-search.ruler-point-process )
pods_install_executables( object-search.ruler-point-process-test-gem-k-tick-process )
