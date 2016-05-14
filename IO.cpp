#include <iostream>
#include "Ellipse.h"

void EllipseGenerator::choose_geometry()
{
  // Choose center
  _geometry.center(0) = _center_dist(_engine);
  _geometry.center(1) = _center_dist(_engine);
  
  // Choose radius
  _geometry.radius(0) = _radius_dist(_engine);
  _geometry.radius(1) = _radius_dist(_engine);

  // Choose rotation
  _geometry.rotation = _angle_dist(_engine);
  _rotation = Eigen::Rotation2Df(_geometry.rotation);
  
  // Choose arc span
  {
    float phiMin, phiMax;
    do {
      phiMin = _angle_dist(_engine);
      phiMax = _angle_dist(_engine);
      if (phiMax < phiMin)
        std::swap(phiMax, phiMin);
    } while (phiMax - phiMin < _min_arc);
    _arc_dist.param(std::uniform_real_distribution<float>::param_type(phiMin, phiMax));
  }
}

Eigen::Vector2f EllipseGenerator::operator()()
{
  float phi = _arc_dist(_engine);
  Eigen::Array2f circle_point(std::cos(phi), std::sin(phi));
  Eigen::Array2f ellipse_point = _geometry.radius.array() * circle_point;
  Eigen::Array2f noise(_noise_dist(_engine), _noise_dist(_engine));
  Eigen::Vector2f rotated_ellipse_point = _rotation * ellipse_point.matrix();
  return rotated_ellipse_point + _geometry.center;
}

std::tuple<std::vector<Eigen::Vector2f>, EllipseGeometry>
EllipseGenerator::generate(size_t n)
{
  choose_geometry();
  std::vector<Eigen::Vector2f> ret(n);
  std::generate(ret.begin(), ret.end(), *this);
  return std::make_tuple(std::move(ret), _geometry);
}

/////////////////////////////////////////////////////////////////////////////
