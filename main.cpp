#include <cstdlib>
#include <iostream>
#include <opencv/cv.hpp>
#include "Ellipse.h"

static constexpr int MAX_CENTER = 600;

static std::tuple<EllipseGeometry, Eigen::MatrixX2f> generate_problem(size_t n);
static void plot(const Eigen::MatrixX2f& points, const EllipseGeometry& geometry);

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

  auto conic = fit_solver(get<1>(problem));
  cout << "COEFFICIENTS:\n" << get<0>(conic)
      << "\nOFFSET:\n" << get<1>(conic)
      << endl;
  
  auto ell = to_ellipse(conic);
  cout << "FIT:\n" << ell << endl;
  
  plot(get<1>(problem), ell);
  
  return 0;
}

// We hard-code many of the parameters.
static std::tuple<EllipseGeometry, Eigen::MatrixX2f> generate_problem(size_t n)
{
  auto g = get_ellipse_generator(MAX_CENTER, 2*M_PI/16, 2, Eigen::Vector2f(40, 500), 0.9);
  Eigen::MatrixX2f ret(n, 2);
  for (size_t i = 0; i < n; ++i)
    ret.row(i) = g();
  return std::make_tuple(g.geometry(), ret);
}

static void plot(const Eigen::MatrixX2f& points, const EllipseGeometry& geometry)
{
  using namespace cv;

  namedWindow("FITTING", WINDOW_AUTOSIZE);
  Mat image(MAX_CENTER+100, MAX_CENTER+100, CV_8UC1);
  
  for (size_t i = 0; i < points.rows(); ++i)
    circle(image, Point(points(i,0), points(i,1)), 4, 255);
  
  ellipse(image, Point(geometry.center(0), geometry.center(1)), Size(geometry.radius(0), geometry.radius(1)),
      geometry.angle*180/M_PI, 0, 360, 255);
  
  imshow("FITTING", image);
  waitKey(0);
}
