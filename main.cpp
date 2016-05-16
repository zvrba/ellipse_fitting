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

  auto conic = fit_solver(get<1>(problem));
  cout << "COEFFICIENTS:\n" << get<0>(conic)
      << "\nOFFSET:\n" << get<1>(conic)
      << endl;
  
  auto ell = to_ellipse(conic);
  auto cv_ell = cv_fit_ellipse(get<1>(problem));
  
  cout << "FIT: " << ell << endl;
  cout << "CV FIT: " << cv_ell << endl;
  
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
  Mat image(MAX_SIZE, MAX_SIZE, CV_8UC1);
  
  for (size_t i = 0; i < points.rows(); ++i)
    circle(image, Point(points(i,0), points(i,1)), 4, 255);
  
  ellipse(image, Point(eg1.center(0), eg1.center(1)), Size(eg1.radius(0), eg1.radius(1)), eg1.angle*180/M_PI, 0, 360, 255, 2);
  ellipse(image, Point(eg2.center(0), eg2.center(1)), Size(eg2.radius(0), eg2.radius(1)), eg2.angle*180/M_PI, 0, 360, 255, 1);
  
  imshow("FITTING", image);
  waitKey(0);
}

static std::tuple<EllipseGeometry, Eigen::MatrixX2f> generate_cv_fail()
{
  using Eigen::Vector2f;
  EllipseGeometry geom{Vector2f(1094.5, 1225.16), Vector2f(567.041, 365.318), 0.245385};
  Eigen::MatrixX2f data(10, 2);
  data <<
      672.859, 1407.93,
      711.274, 1440.01,
      704.937, 1436.61,
      801.286, 1499.89,
      646.25,  1381.52,
      740.138, 1460.1,
      783.188, 1488.21,
      625.618, 1358.16,
      768.642, 1480.37,
      723.981, 1451.32;
  return std::make_tuple(geom, data);
}
