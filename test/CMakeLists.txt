
add_executable( ruler-point-process-test-likelihood-montecarlo
  test-likelihood-montecarlo.cpp )
pods_use_pkg_config_packages( ruler-point-process-test-likelihood-montecarlo math-core point-process-core probability-core ruler-point-process lcmtypes_math-core lcmtypes_point-process-core lcmtypes_probability-core lcmtypes_igmm-point-process )
pods_install_executables( ruler-point-process-test-likelihood-montecarlo )


add_executable( ruler-point-process-test-mcmc
  test-mcmc.cpp
)
pods_use_pkg_config_packages( ruler-point-process-test-mcmc
  math-core lcmtypes_math-core
  probability-core lcmtypes_probability-core
  point-process-core lcmtypes_point-process-core
  point-process-experiment-core
  ruler-point-process lcmtypes_ruler-point-process
)
pods_install_executables( ruler-point-process-test-mcmc )

  