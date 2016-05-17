#include <cstdlib>
#include <iostream>
#include <string>
#include <functional>
#include <opencv/cv.hpp>
#include "Ellipse.h"

static constexpr size_t MAX_RADIUS = 600;
static constexpr size_t MAX_SIZE = 1920;

static std::tuple<EllipseGeometry, Eigen::MatrixX2f> generate_problem(size_t n, float sigma);
static std::tuple<EllipseGeometry, Eigen::MatrixX2f> generate_cv_fail();
static void output_solution(const std::string& name, const Eigen::MatrixX2f& points, std::function<EllipseGeometry(const Eigen::MatrixX2f&)> solver,
    cv::Mat& image, cv::Scalar color, int thickness);
static void plot(const Eigen::MatrixX2f& points, const EllipseGeometry& eg1, const EllipseGeometry& eg2, const EllipseGeometry& eg3);

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

  const auto& points = get<1>(problem);
  
  cout << "PROBLEM:"
      << "\nELLIPSE:\n" << get<0>(problem)
      << "\nPOINTS:\n" << points
      << endl;
  
  cv::namedWindow("FITTING", cv::WINDOW_AUTOSIZE);
  cv::Mat image(MAX_SIZE, MAX_SIZE, CV_8UC4);
  
  // Plot points: white
  for (size_t i = 0; i < points.rows(); ++i)
    cv::circle(image, cv::Point(points(i,0), points(i,1)), 6, cv::Scalar(255, 255, 255));
  
  output_solution("AL_FIT", points, fit_ellipse, image, cv::Scalar(255, 0, 255), 2);      // algebraic fit; purple
  output_solution("CV_FIT", points, cv_fit_ellipse, image, cv::Scalar(0, 0, 255), 1);     // CV fit: red
  //output_solution("GE_FIT", points, geom_fit_ellipse, image, cv::Scalar(255, 0, 0), 1); // geometric fit: blue

  cv::imshow("FITTING", image);
  cv::waitKey(0);
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

static void output_solution(const std::string& name, const Eigen::MatrixX2f& points, std::function<EllipseGeometry(const Eigen::MatrixX2f&)> solver,
    cv::Mat& image, cv::Scalar color, int thickness)
{
  auto ell = solver(points);
  auto conic = to_conic(ell);
  std::cout << name
      << "\nCOEFFICIENTS:\n" << std::get<0>(conic)
      << "\nELLIPSE: " << ell
      << "\nERROR: " << fit_error(points, conic)
      << std::endl;
  cv::ellipse(image, cv::Point(ell.center(0), ell.center(1)), cv::Size(ell.radius(0), ell.radius(1)), ell.angle*180/M_PI, 0, 360, color, thickness);
}

