#include <cstdlib>
#include <iostream>
#include <string>
#include <opencv/cv.hpp>
#include "Ellipse.h"

static constexpr size_t MAX_RADIUS = 600;
static constexpr size_t MAX_SIZE = 1920;

static std::tuple<EllipseGeometry, Eigen::MatrixX2f> generate_problem(size_t n, float sigma);
static std::tuple<EllipseGeometry, Eigen::MatrixX2f> generate_cv_fail();
static void plot(const Eigen::MatrixX2f& points, const EllipseGeometry& eg1, const EllipseGeometry& eg2);

int main(int argc, char** argv)
{
  using namespace std;
  
  if (argc < 3) {
    cerr << "USAGE: " << argv[0] << " NPOINTS SIGMA" << endl;
    return 1;
  }
  
  std::tuple<EllipseGeometry, Eigen::MatrixX2f> problem;
  
  if (std::string("FAIL") == argv[1]) {
    problem = generate_cv_fail();
  }
  else {
    size_t npoints = std::atoi(argv[1]);
    float sigma = std::atof(argv[2]);
    problem = generate_problem(npoints, sigma);
  }
  
  cout << "PROBLEM:"
      << "\nELLIPSE:\n" << get<0>(problem)
      << "\nPOINTS:\n" << get<1>(problem)
      << endl;

  auto ell = fit_ellipse(get<1>(problem));
  auto conic = to_conic(ell);
  cout << "COEFFICIENTS:\n" << get<0>(conic)
      << "\nELLIPSE: " << ell
      << "\nERROR: " << fit_error(get<1>(problem), conic)
      << endl;
  
  auto cv_ell = cv_fit_ellipse(get<1>(problem));
  auto cv_conic = to_conic(cv_ell);
  cout << "CV COEFFICIENTS:\n" << get<0>(cv_conic)
      << "\nCV ELLIPSE: " << cv_ell
      << "\nCV ERROR: " << fit_error(get<1>(problem), cv_conic)
      << endl;
  
  plot(get<1>(problem), ell, cv_ell);
  
  return 0;
}

// We hard-code many of the parameters.
static std::tuple<EllipseGeometry, Eigen::MatrixX2f> generate_problem(size_t n, float sigma)
{
  using Eigen::Vector2f;
  auto g = get_ellipse_generator(Vector2f(MAX_SIZE/4, MAX_SIZE-MAX_SIZE/4), Vector2f(50, MAX_RADIUS), 0.1, 2*M_PI/32, sigma);
  Eigen::MatrixX2f ret(n, 2);
  for (size_t i = 0; i < n; ++i)
    ret.row(i) = g();
  return std::make_tuple(g.geometry(), ret);
}

static void plot(const Eigen::MatrixX2f& points, const EllipseGeometry& eg1, const EllipseGeometry& eg2)
{
  using namespace cv;

  namedWindow("FITTING", WINDOW_AUTOSIZE);
  Mat image(MAX_SIZE, MAX_SIZE, CV_8UC4);
  
  for (size_t i = 0; i < points.rows(); ++i)
    circle(image, Point(points(i,0), points(i,1)), 6, Scalar(255, 255, 255));
  
  ellipse(image, Point(eg1.center(0), eg1.center(1)), Size(eg1.radius(0), eg1.radius(1)), eg1.angle*180/M_PI, 0, 360, Scalar(255, 0, 255), 2);
  ellipse(image, Point(eg2.center(0), eg2.center(1)), Size(eg2.radius(0), eg2.radius(1)), eg2.angle*180/M_PI, 0, 360, Scalar(0, 0, 255), 1);
  
  imshow("FITTING", image);
  waitKey(0);
}

static std::tuple<EllipseGeometry, Eigen::MatrixX2f> generate_cv_fail()
{
  using Eigen::Vector2f;
  EllipseGeometry geom{Vector2f(1094.5, 1225.16), Vector2f(567.041, 365.318), 0.245385};
  Eigen::MatrixX2f data(10, 2);
  data <<
      924.784, 764.160,
      928.388, 615.903,
      847.4  , 888.014,
      929.406, 741.675,
      904.564, 825.605,
      926.742, 760.746,
      863.479, 873.406,
      910.987, 808.863,
      929.145, 744.976,
      917.474, 791.823;
  return std::make_tuple(geom, data);
}
