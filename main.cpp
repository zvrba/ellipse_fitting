#include <cstdlib>
#include <iostream>
#include "Ellipse.h"

static std::tuple<EllipseGeometry, Eigen::MatrixX2f> generate_problem(size_t n);

int main(int argc, char** argv)
{
  using namespace std;
  
  if (argc < 3) {
    cerr << "USAGE: " << argv[0] << " NPOINTS SIGMA" << endl;
    return 1;
  }
  
  size_t npoints = std::atoi(argv[1]);
  float sigma = std::atof(argv[2]);
  
  auto problem = generate_problem(npoints);
  cout << "PROBLEM:"
      << "\nELLIPSE:\n" << get<0>(problem)
      << "\nPOINTS:\n" << get<1>(problem)
      << endl;

  auto solution = fit_solver(get<1>(problem));
  cout << "COEFFICIENTS:\n" << get<0>(solution)
      << "\nCENTER:\n" << get<1>(solution)
      << endl;
  
  return 0;
}

// We hard-code many of the parameters.
static std::tuple<EllipseGeometry, Eigen::MatrixX2f> generate_problem(size_t n)
{
  auto g = get_ellipse_generator(2048, 2*M_PI/32, 2, Eigen::Vector2f(40, 500), 0.9);
  Eigen::MatrixX2f ret(n, 2);
  for (size_t i = 0; i < n; ++i)
    ret.row(i) = g();
  return std::make_tuple(g.geometry(), ret);
}

