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
  float rotation;
};

class EllipseGenerator
{
private:
  // Generator parameters
  const float _min_arc;
  const float _min_ratio;
  std::ranlux24 _engine;
  std::uniform_real_distribution<float> _center_dist;
  std::uniform_real_distribution<float> _radius_dist;
  std::uniform_real_distribution<float> _angle_dist;
  std::normal_distribution<float> _noise_dist;

  // Per-ellipse parameters
  EllipseGeometry _geometry;
  Eigen::Rotation2Df _rotation;
  std::uniform_real_distribution<float> _arc_dist;
  
  void choose_geometry();
  
public:
  EllipseGenerator(float maxCenter, float minArc, float sigma, float minRadius, float maxRadius, float minRatio) :
    _min_arc(minArc),
    _min_ratio(minRatio),
    _engine(2718282),
    _center_dist(0, maxCenter),
    _radius_dist(minRadius, maxRadius),
    _angle_dist(0, 2*M_PI),
    _noise_dist(0, sigma)
  { }
    
  Eigen::Vector2f operator()();
  std::tuple<std::vector<Eigen::Vector2f>, EllipseGeometry> generate(size_t n);
};