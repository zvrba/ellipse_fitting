#include <cmath>
#include <cfloat>
#include "Ellipse.h"

// Taken from OpenCV old code; see
// https://github.com/Itseez/opencv/commit/4eda1662aa01a184e0391a2bb2e557454de7eb86#diff-97c8133c3c171e64ea0df0db4abd033c
EllipseGeometry to_ellipse(const Conic& conic)
{
  using namespace Eigen;
  auto coef = std::get<0>(conic);

  float idet = coef(0)*coef(2) - coef(1)*coef(1)/4; // ac-b^2/4
  idet = idet > FLT_EPSILON ? 1.f/idet : 0;
  
  float scale = std::sqrt(idet/4);
  if (scale < FLT_EPSILON)
    throw std::domain_error("to_ellipse_2: singularity 1");
  
  coef *= scale;
  float aa = coef(0), bb = coef(1), cc = coef(2), dd = coef(3), ee = coef(4), ff = coef(5);
  
  const Vector2f c = Vector2f(-dd*cc + ee*bb/2, -aa*ee + dd*bb/2) * 2;
  
  // offset ellipse to (x0,y0)
  ff += aa*c(0)*c(0) + bb*c(0)*c(1) + cc*c(1)*c(1) + dd*c(0) + ee*c(1);
  if (std::fabs(ff) < FLT_EPSILON)
    throw std::domain_error("to_ellipse_2: singularity 2");
  
  Matrix2f S;
  S << aa, bb/2, bb/2, cc;
  S /= -ff;

  // SVs are sorted from largest to smallest
  JacobiSVD<Matrix2f> svd(S, ComputeFullU);
  const auto& vals = svd.singularValues();
  const auto& mat_u = svd.matrixU();

  Vector2f center = c + std::get<1>(conic);
  Vector2f radius = Vector2f(std::sqrt(1.f/vals(0)), std::sqrt(1.f/vals(1)));
  float angle = M_PI - std::atan2(mat_u(0,1), mat_u(1,1));
  return EllipseGeometry{center, radius, angle};
}

Conic to_conic(const EllipseGeometry& eg)
{
  const float s = std::sin(eg.angle);
  const float c = std::cos(eg.angle);
  const float aa = eg.radius(0), bb = eg.radius(1);
  const float hh = eg.center(0), kk = eg.center(1);
  const auto sq = [](float x) { return x*x; };

  Eigen::Vector6f coef;

  coef(0) = sq(bb*c) + sq(aa*s);
  coef(1) = -2*c*s*(sq(aa) - sq(bb));
  coef(2) = sq(bb*s) + sq(aa*c);
  coef(3) = -2*coef(0)*hh - kk*coef(1);
  coef(4) = -2*coef(2)*kk - hh*coef(1);
  coef(5) = -sq(aa*bb) + coef(0)*sq(hh) + coef(1)*hh*kk + coef(2)*sq(kk);
  return std::make_tuple(coef, Eigen::Vector2f(0, 0));
}

float fit_error(const Eigen::MatrixX2f& points, const Conic& conic)
{
  using namespace Eigen;
  
  if (std::get<1>(conic) != Vector2f(0,0))
    throw std::domain_error("fit_error: offset");
  
  const auto& coef = std::get<0>(conic);
  const auto point_ev = [&](const Vector2f& p) {
    float ev = coef(0)*p(0)*p(0) + coef(1)*p(0)*p(1) + coef(2)*p(1)*p(1) + coef(3)*p(0) + coef(4)*p(1) + coef(5);
    return ev*ev;
  };
  
  float err_sq = 0;
  for (size_t i = 0; i < points.rows(); ++i)
    err_sq += point_ev(points.row(i));
  return std::sqrt(err_sq);
}
