#include <cmath>
#include "Ellipse.h"

static std::ranlux24 G_engine(271828);

std::ostream& operator<<(std::ostream& os, const EllipseGeometry& eg)
{
  os << "{ C:" << eg.center.transpose() << "; R:" << eg.radius.transpose() << "; A:" << eg.radius << " }";
  return os;
}

Eigen::Vector2f EllipseGenerator::operator()()
{
  float phi = _arc_dist(G_engine);
  Eigen::Array2f circle_point(std::cos(phi), std::sin(phi));
  Eigen::Array2f ellipse_point = _geometry.radius.array() * circle_point;
  Eigen::Vector2f noise(_noise_dist(G_engine), _noise_dist(G_engine));
  Eigen::Vector2f rotated_ellipse_point = _rotation * (ellipse_point.matrix() + noise);
  return rotated_ellipse_point + _geometry.center;
}

// eccentricity: 0->circle, limit->1: line
EllipseGenerator get_ellipse_generator(float max_center, float min_arc_angle, float sigma,
    Eigen::Vector2f radiusSpan, float max_eccentricity)
{
  std::uniform_real_distribution<float> center_dist(0, max_center);
  std::uniform_real_distribution<float> radius_dist(radiusSpan(0), radiusSpan(1));
  std::uniform_real_distribution<float> angle_dist(0, 2*M_PI);
  
  // Center.
  float cx = center_dist(G_engine);
  float cy = center_dist(G_engine);
  
  // Rotation.
  float angle = angle_dist(G_engine);
  
  // Radii; eccentricity must not be below min_eccentricity.
  float a, b;
  do {
    a = radius_dist(G_engine);
    b = radius_dist(G_engine);
    if (a < b)
      std::swap(a, b);
  } while (std::sqrt(a*a-b*b) >= max_eccentricity);
  
  // Arc span; must be at least min_arc_angle
  float phi_min, phi_max;
  do {
    phi_min = angle_dist(G_engine);
    phi_max = angle_dist(G_engine);
    if (phi_max < phi_min)
      std::swap(phi_max, phi_min);
  } while (phi_max - phi_min < min_arc_angle);
  
  EllipseGeometry geometry{Eigen::Vector2f(cx, cy), Eigen::Vector2f(a, b), angle};
  return EllipseGenerator(geometry, Eigen::Vector2f(phi_min, phi_max), sigma);
}

// We hard-code many of the parameters.
std::tuple<EllipseGeometry, Eigen::MatrixX2f> generate_problem(size_t n)
{
  auto g = get_ellipse_generator(2048, 2*M_PI/32, 2, Eigen::Vector2f(40, 500), 0.9);
  Eigen::MatrixX2f ret(n, 2);
  for (size_t i = 0; i < n; ++i)
    ret.row(i) = g();
  return std::make_tuple(g.geometry(), ret);
}
