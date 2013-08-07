
#include <ruler-point-process/mcmc.hpp>
#include <math-core/matrix.hpp>
#include <math-core/io.hpp>
#include <probability-core/distribution_utils.hpp>
#include <iostream>


using namespace probability_core;
using namespace ruler_point_process;
using namespace math_core;


int main()
{

  // Ok, create some distribution of a fake mixture
  int dim = 1;
  gaussian_distribution_t spread;
  spread.dimension = 1;
  spread.means.push_back( 0.0 );
  spread.covariance = to_dense_mat(Eigen::MatrixXd::Identity(dim,dim) * 1.0 );
  gamma_distribution_t period;
  period.shape = 2;
  period.rate = 0.5;
  gamma_distribution_t length;
  length.shape = 2;
  length.rate = 0.5;
  gaussian_distribution_t start;
  start.dimension = 1;
  start.means.push_back( 0 );
  start.covariance = to_dense_mat( Eigen::MatrixXd::Identity(dim,dim) * 1.0 );
  gaussian_distribution_t direction;
  direction.dimension = 1;
  direction.means.push_back( 10 );
  direction.covariance = to_dense_mat( Eigen::MatrixXd::Identity(dim,dim) * 1.0 );

  // some negative observations
  std::vector<nd_aabox_t> negative_observations;
  nd_aabox_t reg0 = { 1, point(0), point(1) };
  negative_observations.push_back( reg0 );
  
  // ok, now get a point for the likelihood
  nd_point_t p = point( 10 );
  
  double lik = likelihood_of_single_point_for_mixture( p,
						       negative_observations,
						       spread,
						       period,
						       length,
						       start,
						       direction );
  std::cout << "Lik( " << p << " ) = " << lik << std::endl;

  return 0;
}
