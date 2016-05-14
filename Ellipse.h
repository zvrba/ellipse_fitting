#pragma once
#include <random>
#include <vector>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>

class EllipseGenerator
{
public:
  struct Parameters
  {
    Eigen::Vector2f center;
    Eigen::Vector2f radius;
    Eigen::Vector2f arc_span;
    float rotation;
  };
  
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
  Parameters _parameters;
  Eigen::Rotation2Df _rotation;
  std::uniform_real_distribution<float> _arc_dist;
  
  void choose_parameters();
  
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
  std::tuple<std::vector<Eigen::Vector2f>, Parameters> generate(size_t n);
};