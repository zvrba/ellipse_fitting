#include <iostream>
#include "Ellipse.h"

void EllipseGenerator::choose_parameters()
{
  // Choose center
  _parameters.center(0) = _center_dist(_engine);
  _parameters.center(1) = _center_dist(_engine);
  
  // Choose radius
  _parameters.radius(0) = _radius_dist(_engine);
  _parameters.radius(1) = _radius_dist(_engine);
  
  // Choose arc span
  {
    float phiMin, phiMax;
    do {
      phiMin = _angle_dist(_engine);
      phiMax = _angle_dist(_engine);
      if (phiMax < phiMin)
        std::swap(phiMax, phiMin);
    } while (phiMax - phiMin < _min_arc);
    _parameters.arc_span = Eigen::Vector2f(phiMin, phiMax);
  }
  
  // Choose rotation
  _parameters.rotation = _angle_dist(_engine);
  _rotation = Eigen::Rotation2Df(_parameters.rotation);
  
  // Set arc RNG parameters
  _arc_dist.param(std::uniform_real_distribution<float>::param_type(
    _parameters.arc_span(0), _parameters.arc_span(1)));
}

Eigen::Vector2f EllipseGenerator::operator()()
{
  float phi = _arc_dist(_engine);
  Eigen::Array2f circle_point(std::cos(phi), std::sin(phi));
  Eigen::Array2f ellipse_point = _parameters.radius.array() * circle_point;
  Eigen::Array2f noise(_noise_dist(_engine), _noise_dist(_engine));
  Eigen::Vector2f rotated_ellipse_point = _rotation * ellipse_point.matrix();
  return rotated_ellipse_point + _parameters.center;
}

std::tuple<std::vector<Eigen::Vector2f>, EllipseGenerator::Parameters>
EllipseGenerator::generate(size_t n)
{
  choose_parameters();
  std::uniform_real_distribution<float> arc_dist(_parameters.arc_span(0), _parameters.arc_span(1));
  std::vector<Eigen::Vector2f> ret(n);
  std::generate(ret.begin(), ret.end(), *this);
  return std::make_tuple(std::move(ret), _parameters);
}

/////////////////////////////////////////////////////////////////////////////
