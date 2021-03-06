#pragma once
#include <iostream>
#include <random>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace Eigen
{
using Vector6f = Eigen::Matrix<float, 6, 1>;
}

struct EllipseGeometry
{
  Eigen::Vector2f center;
  Eigen::Vector2f radius;
  float angle;
  
  friend std::ostream& operator<<(std::ostream&, const EllipseGeometry&);
};

// 6 coefficients + center offset
using Conic = std::tuple<Eigen::Vector6f, Eigen::Vector2f>;

class EllipseGenerator
{
  const EllipseGeometry _geometry;
  Eigen::Rotation2Df _rotation;
  std::uniform_real_distribution<float> _arc_dist;
  std::normal_distribution<float> _noise_dist;
public:
  EllipseGenerator(const EllipseGeometry& geometry, Eigen::Vector2f arc_span, float sigma) :
      _geometry(geometry), _rotation(_geometry.angle), _arc_dist(arc_span(0), arc_span(1)), _noise_dist(0, sigma)
  { }
  const EllipseGeometry& geometry() const { return _geometry; }
  Eigen::Vector2f operator()();
};

EllipseGenerator get_ellipse_generator(Eigen::Vector2f center_span, Eigen::Vector2f radius_span, float min_r_ratio,
    float min_arc_angle, float sigma);

EllipseGeometry fit_ellipse(const Eigen::MatrixX2f& points);
EllipseGeometry geom_fit_ellipse(const Eigen::MatrixX2f& points);
EllipseGeometry cv_fit_ellipse(const Eigen::MatrixX2f& points);
float fit_error(const Eigen::MatrixX2f& points, const Conic& conic);

Eigen::Vector2f get_offset(const Eigen::MatrixX2f& points);
EllipseGeometry to_ellipse(const Conic&);
Conic to_conic(const EllipseGeometry& eg);
