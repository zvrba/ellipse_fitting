#pragma once
#include <random>
#include <vector>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>

struct EllipseGeometry
{
  Eigen::Vector2f center;
  Eigen::Vector2f radius;
  float angle;
};

class EllipseGenerator
{
  const EllipseGeometry _geometry;
  Eigen::Rotation2Df _rotation;
  std::uniform_real_distribution<float> _arc_dist;
  std::normal_distribution<float> _noise_dist;
public:
  EllipseGenerator(const EllipseGeometry& geometry, Eigen::Vector2f arc_span, float sigma) :
    _geometry(geometry), _rotation(_geometry.angle), _arc_dist(arc_span(0), arc_span(1)), _noise_dist(sigma)
  { }
  const EllipseGeometry& geometry() const { return _geometry; }
  Eigen::Vector2f operator()();
};

EllipseGenerator get_ellipse_generator(float max_center, float min_arc_angle, float sigma,
    Eigen::Vector2f radiusSpan, float min_eccentricity);

