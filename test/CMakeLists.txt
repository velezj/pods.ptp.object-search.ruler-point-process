
add_executable( p2l-ruler-point-process-test-likelihood-montecarlo
  test-likelihood-montecarlo.cpp )
pods_use_pkg_config_packages( p2l-ruler-point-process-test-likelihood-montecarlo
    p2l-math-core 
    p2l-point-process-core 
    p2l-probability-core 
    p2l-ruler-point-process 
    lcmtypes_p2l-math-core 
    lcmtypes_p2l-probability-core 
    lcmtypes_p2l-igmm-point-process )
pods_install_executables( p2l-ruler-point-process-test-likelihood-montecarlo )


add_executable( p2l-ruler-point-process-test-mcmc
  test-mcmc.cpp
)
pods_use_pkg_config_packages( p2l-ruler-point-process-test-mcmc
  p2l-math-core lcmtypes_p2l-math-core
  p2l-probability-core lcmtypes_p2l-probability-core
  p2l-point-process-core
  p2l-point-process-experiment-core
  p2l-ruler-point-process
)
pods_install_executables( p2l-ruler-point-process-test-mcmc )

  
