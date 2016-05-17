#include "Ellipse.h"

static std::mt19937_64 G_engine(time(0));//(271828);

std::ostream& operator<<(std::ostream& os, const EllipseGeometry& eg)
{
  os << "{ C:" << eg.center.transpose() << "; R:" << eg.radius.transpose() << "; A:" << eg.angle << " }";
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
EllipseGenerator get_ellipse_generator(Eigen::Vector2f center_span, Eigen::Vector2f radius_span, float min_r_ratio,
    float min_arc_angle, float sigma)
{
  std::uniform_real_distribution<float> center_dist(center_span(0), center_span(1));
  std::uniform_real_distribution<float> radius_dist(radius_span(0), radius_span(1));
  std::uniform_real_distribution<float> angle_dist(0, 2*M_PI);
  
  // Center.
  float cx = center_dist(G_engine);
  float cy = center_dist(G_engine);
  
  // Rotation. Doesn't make sense to rotate ellipse more than 180 deg.
  float angle = angle_dist(G_engine);
  if (angle > M_PI)
    angle -= M_PI;
  
  // Radii; ratio of minor/major radii must not be < min_r_ratio
  float a, b;
  do {
    a = radius_dist(G_engine);
    b = radius_dist(G_engine);
    if (a < b)
      std::swap(a, b);
  } while (b/a < min_r_ratio);
  
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

