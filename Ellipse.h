#pragma once
#include <random>
#include <vector>
#include <cmath>
#include <Eigen/Core>

class EllipseGenerator
{
  float _minArc;
  float _minRatio;
  std::ranlux24 _engine;
  std::uniform_real_distribution<float> _centerDist;
  std::uniform_real_distribution<float> _radiusDist;
  std::uniform_real_distribution<float> _angleDist;
  std::normal_distribution<float> _noiseDist;
  
public:
  EllipseGenerator(float maxCenter, float minArc, float sigma, float minRadius, float maxRadius, float minRatio) :
    _minArc(minArc),
    _minRatio(minRatio),
    _engine(2718282),
    _centerDist(0, maxCenter),
    _radiusDist(minRadius, maxRadius),
    _angleDist(0, 2*M_PI),
    _noiseDist(0, sigma)
  { }
  
  Eigen::Vector2f operator()
  {
    
  }
};