#include <cmath>
#include <vector>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include "Ellipse.h"


// Delegates fitting to OpenCV.
EllipseGeometry cv_fit_ellipse(const Eigen::MatrixX2f& points)
{
  std::vector<cv::Point2f> cvp(points.rows());
  for (size_t i = 0; i < points.rows(); ++i)
    cvp[i] = cv::Point2f(points(i,0), points(i,1));
  auto rr = cv::fitEllipse(cvp);
  Eigen::Vector2f center(rr.center.x, rr.center.y);
  Eigen::Vector2f radius(rr.size.width/2, rr.size.height/2);
  float angle = rr.angle * M_PI / 180;
  return EllipseGeometry{center, radius, angle};
}

/////////////////////////////////////////////////////////////////////////////
// FITTING based on the following paper: http://autotrace.sourceforge.net/WSCG98.pdf

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

static Conic fit_solver(const Eigen::MatrixX2f& points)
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
    Vector3f a1 = M_ev.eigenvectors().real().col(imin);
    Vector3f a2 = T*a1;
    ret.block<3,1>(0,0) = a1;
    ret.block<3,1>(3,0) = a2;
  }
  return std::make_tuple(ret, offset);
}

EllipseGeometry fit_ellipse(const Eigen::MatrixX2f& points)
{
  return to_ellipse(fit_solver(points));
}

