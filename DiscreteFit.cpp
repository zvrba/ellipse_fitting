#include <stdexcept>
#include <float.h>
#include "Ellipse.h"

// Implementation of the paper
// ElliFit: An unconstrained, non-iterative, least squares based geometric Ellipse Fitting method
// by Prasad, Quek in Pattern Recognition, January 2012

namespace Eigen
{
using Vector5f = Matrix<float, 5, 1>;
}

using PhiOffset = std::tuple<Eigen::Vector5f, Eigen::Vector2f>;

static PhiOffset get_phi_offset(const Eigen::MatrixX2f& points)
{
  using namespace Eigen;
  const auto offset = get_offset(points);
  const size_t n = points.rows();
  size_t zero_count = 0;
  
  Matrix<float, Dynamic, 5> X(n, 5);
  VectorXf Y(n);

  for (size_t i = 0; i < n; ++i) {
    auto p = points.row(i).transpose() - offset;
    X(i,0) = -p(0)*p(0);
    X(i,1) = -p(0)*p(1);
    X(i,2) = p(0);
    X(i,3) = p(1);
    X(i,4) = 1;
    Y(i) = p(1) * p(1);
    zero_count += p(1) < 2*FLT_EPSILON;
  }
  
  if (n - zero_count < 5)
    throw std::domain_error("geom_fit_ellipse: too few points");
  
  // XXX: Should be computed directly, see the paper
  auto phi = (X.transpose()*X).inverse() * X.transpose() * Y;
  return std::make_tuple(phi, offset);
}

static Eigen::Vector5f get_a(const PhiOffset& phi_offset)
{
  const auto& phi = std::get<0>(phi_offset);
  const auto& offset = std::get<1>(phi_offset);
  const float f1 = phi(0), f2 = phi(1), f3 = phi(2), f4 = phi(3), f5 = phi(4);
  Eigen::Vector5f A;
  
  const float disc = f2*f2 - 4*f1;
  {
    float nom = f2*f3*f4 - f4*f4*f1 - f3*f3 - f5*disc;
    float den1 = std::sqrt((1-f1)*(1-f1) + f2*f2);
    A(0) = 2*std::sqrt(nom / (disc*(1+f1-den1)));
    A(1) = 2*std::sqrt(nom / (disc*(1+f1+den1)));
  }
  
  A(2) = -std::atan2(-f2, 1-f1)/2;
  A(3) = (f2*f4 - 2*f3) / disc;
  A(4) = (f2*f3 - 2*f1*f4) / disc;
  
  A(3) += offset(0);
  A(4) += offset(1);
  
  return A;
}

EllipseGeometry geom_fit_ellipse(const Eigen::MatrixX2f& points)
{
  using Eigen::Vector2f;
  auto phi_offset = get_phi_offset(points);
  auto A = get_a(phi_offset);
  return EllipseGeometry{Vector2f(A(3),A(4)), Vector2f(A(0),A(1)), A(2) };
}
