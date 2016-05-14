#include <iostream>

static std::tuple<EllipseGeometry, Eigen::MatrixX2f> generate_problem(size_t n);

int main(int argc, char** argv)
{
  if (argc < 3) {
    std::cerr << "USAGE: " << argv[0] << " NPOINTS SIGMA";
    return 1;
  }
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

