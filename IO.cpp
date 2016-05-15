#include <cmath>
#include <Eigen/Eigenvalues>
#include <float.h>
#include "Ellipse.h"

static std::ranlux24 G_engine(271828);

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
EllipseGenerator get_ellipse_generator(float max_center, float min_arc_angle, float sigma,
    Eigen::Vector2f radiusSpan, float max_eccentricity)
{
  std::uniform_real_distribution<float> center_dist(0, max_center);
  std::uniform_real_distribution<float> radius_dist(radiusSpan(0), radiusSpan(1));
  std::uniform_real_distribution<float> angle_dist(0, 2*M_PI);
  
  // Center.
  float cx = center_dist(G_engine);
  float cy = center_dist(G_engine);
  
  // Rotation.
  float angle = angle_dist(G_engine);
  
  // Radii; eccentricity must not be below min_eccentricity.
  float a, b;
  do {
    a = radius_dist(G_engine);
    b = radius_dist(G_engine);
    if (a < b)
      std::swap(a, b);
  } while (std::sqrt(a*a-b*b) >= max_eccentricity);
  
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

/////////////////////////////////////////////////////////////////////////////
// FITTING based on the following paper: http://autotrace.sourceforge.net/WSCG98.pdf

static Eigen::Vector2f get_offset(const Eigen::MatrixX2f& points)
{
  auto sum = points.colwise().sum();
  return sum / points.rows();
}

static std::tuple<Eigen::Matrix3f,Eigen::Matrix3f,Eigen::Matrix3f>
get_scatter_matrix(const Eigen::MatrixX2f& points, const Eigen::Vector2f& offset)
{
  using namespace Eigen;
  
  const auto qf = [&](size_t i) {
    Vector2f p = points.row(i);
    auto pc = p - offset;
    return Vector3f(pc(0)*pc(0), pc(0)*pc(1), pc(1)*pc(1));
  };
  const auto lf = [&](size_t i) {
    Vector2f p = points.row(i);
    auto pc = p - offset;
    return Vector3f(pc(0), pc(1), 1);
  };
  
  const size_t n = points.rows();
  MatrixX3f D1(n,3), D2(n,3);
  
  // Construct the quadratic and linear parts.  Doing it in two loops has better cache locality.
  for (size_t i = 0; i < n; ++i)
    D1.row(i) = qf(i);
  for (size_t i = 0; i < n; ++i)
    D2.row(i) = lf(i);
  
  // Construct the three parts of the symmetric scatter matrix.
  auto S1 = D1.transpose() * D1;
  auto S2 = D1.transpose() * D2;
  auto S3 = D2.transpose() * D2;
  return std::make_tuple(S1, S2, S3);
}

std::tuple<Eigen::Vector6f, Eigen::Vector2f>
fit_solver(const Eigen::MatrixX2f& points)
{
  using namespace Eigen;
  using std::get;
  
  static const struct C1_Initializer {
    Matrix3f matrix;
    Matrix3f inverse;
    C1_Initializer()
    {
      matrix <<
          0,  0, 2,
          0, -1, 0,
          2,  0, 0;
      inverse <<
            0,  0, 0.5,
            0, -1,   0,
          0.5,  0,   0; 
    };
  } C1;
  
  const auto offset = get_offset(points);
  const auto St = get_scatter_matrix(points, offset);
  const auto& S1 = std::get<0>(St);
  const auto& S2 = std::get<1>(St);
  const auto& S3 = std::get<2>(St);
  const auto T = -S3.inverse() * S2.transpose();
  const auto M = C1.inverse * (S1 + S2*T);
  
  EigenSolver<Matrix3f> M_ev(M);
  Vector3f cond;
  {
    const auto evr = M_ev.eigenvectors().real().array();
    cond = 4*evr.row(0)*evr.row(2) - evr.row(1)*evr.row(1);
  }

  float min = FLT_MAX;
  int imin = -1;
  for (int i = 0; i < 3; ++i)
  if (cond(i) > 0 && cond(i) < min) {
    imin = i; min = cond(i);
  }
  
  Vector6f ret = Matrix<float, 6, 1>::Zero();
  if (imin >= 0) {
    Vector3f a1 = M_ev.eigenvectors().real().row(imin);
    Vector3f a2 = T*a1;
    ret.block<3,1>(0,0) = a1;
    ret.block<3,1>(3,0) = a2;
  }
  return std::make_tuple(ret, offset);
}

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
  
  SelfAdjointEigenSolver<Matrix2f> es(S);
  
  Vector2f center = c + std::get<1>(conic);
  Vector2f radius = Vector2f(std::sqrt(1.f/es.eigenvalues()(0)), std::sqrt(1.f/es.eigenvalues()(1)));
  float angle = M_PI - std::atan2(es.eigenvectors()(1,0), es.eigenvectors()(1,1));
  return EllipseGeometry{center, radius, angle};
}
