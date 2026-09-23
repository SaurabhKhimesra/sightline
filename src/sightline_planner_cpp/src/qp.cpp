#include "sightline_planner/qp.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Cholesky>
#include <Eigen/QR>

namespace sightline {
namespace qp {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

}  // namespace

// The factorisation is recomputed whenever the active set changes instead of being
// updated with Givens rotations. On six variables a QR costs a few microseconds, and
// the bookkeeping the update needs is where that algorithm usually goes wrong.
Solution solve(const MatX &G, const VecX &a, const MatX &C_in, const VecX &b_in,
               int max_iter, double tol) {
  const int n = static_cast<int>(G.rows());
  MatX C = C_in;
  VecX b = b_in;
  if (C.size() == 0) {
    C = MatX::Zero(n, 0);
    b = VecX::Zero(0);
  }
  const int m = static_cast<int>(C.cols());

  const MatX L = Eigen::LLT<MatX>(G).matrixL();          // G = L L'
  const MatX J = L.transpose().triangularView<Eigen::Upper>().solve(MatX::Identity(n, n));
  VecX x = -(J * (J.transpose() * a));                   // unconstrained minimum: dual feasible
  std::vector<int> active;
  VecX u(0);

  for (int iteration = 1; iteration <= max_iter; ++iteration) {
    VecX slack = C.transpose() * x - b;
    for (int k : active) {
      slack[k] = std::max(slack[k], 0.0);                // tight by construction
    }
    int p = -1;
    if (m > 0) {
      slack.minCoeff(&p);
    }
    if (m == 0 || slack[p] >= -tol) {
      VecX full = VecX::Zero(std::max(m, 0));
      for (size_t k = 0; k < active.size(); ++k) {
        full[active[k]] = u[static_cast<int>(k)];
      }
      return Solution{x, active, full, iteration, true};
    }

    const VecX n_p = C.col(p);
    VecX grown(u.size() + 1);
    grown.head(u.size()) = u;
    grown[u.size()] = 0.0;
    u = grown;
    bool added = false;

    for (int inner = 0; inner < max_iter; ++inner) {
      const int q = static_cast<int>(active.size());
      VecX r = VecX::Zero(std::max(q, 0));
      VecX z = VecX::Zero(n);
      if (q > 0) {
        MatX B(n, q);
        for (int k = 0; k < q; ++k) {
          B.col(k) = J.transpose() * C.col(active[k]);
        }
        const Eigen::HouseholderQR<MatX> qr(B);
        const MatX Q = qr.householderQ();                // n by n, the complete factor
        const MatX R = qr.matrixQR().topRows(q).triangularView<Eigen::Upper>();
        const MatX star = J * Q;
        const VecX d = star.transpose() * n_p;
        r = R.topLeftCorner(q, q).triangularView<Eigen::Upper>().solve(d.head(q));
        z = star.rightCols(n - q) * d.tail(n - q);
      } else {
        z = J * (J.transpose() * n_p);
      }

      // how far the dual can go before an active constraint would turn negative
      double t1 = kInf;
      int blocking = -1;
      for (int k = 0; k < q; ++k) {
        if (r[k] > tol) {
          const double ratio = u[k] / r[k];
          if (ratio < t1) {
            t1 = ratio;
            blocking = k;
          }
        }
      }

      const double zz = n_p.dot(z);
      const double t2 = zz > tol ? -slack[p] / zz : kInf;
      const double t = std::min(t1, t2);
      if (!std::isfinite(t)) {
        return Solution{x, active, VecX::Zero(std::max(m, 0)), iteration, false};
      }

      if (std::isfinite(t2)) {
        x = x + t * z;
      }
      if (q > 0) {
        u.head(q) -= t * r;
      }
      u[u.size() - 1] += t;

      if (t2 <= t1 + tol) {                              // the new constraint is now tight
        active.push_back(p);
        added = true;
        break;
      }
      // otherwise an old constraint has hit zero: drop it and try again
      std::vector<int> kept;
      VecX kept_u(q);
      int written = 0;
      for (int k = 0; k < q; ++k) {
        if (k == blocking) {
          continue;
        }
        kept.push_back(active[k]);
        kept_u[written++] = u[k];
      }
      VecX next(written + 1);
      next.head(written) = kept_u.head(written);
      next[written] = u[u.size() - 1];
      active = kept;
      u = next;
      slack = C.transpose() * x - b;
    }
    if (!added) {
      return Solution{x, active, VecX::Zero(std::max(m, 0)), iteration, false};
    }
  }

  VecX full = VecX::Zero(std::max(m, 0));
  for (size_t k = 0; k < active.size(); ++k) {
    full[active[k]] = u[static_cast<int>(k)];
  }
  return Solution{x, active, full, max_iter, true};
}

KktError kkt_error(const MatX &G, const VecX &a, const MatX &C, const VecX &b,
                   const Solution &sol) {
  KktError out;
  const VecX slack = C.size() ? VecX(C.transpose() * sol.x - b) : VecX(0);
  out.stationarity = C.size()
      ? (G * sol.x + a - C * sol.multipliers).cwiseAbs().maxCoeff()
      : (G * sol.x + a).cwiseAbs().maxCoeff();
  out.primal = slack.size() ? std::max(0.0, -slack.minCoeff()) : 0.0;
  out.dual = sol.multipliers.size() ? std::max(0.0, -sol.multipliers.minCoeff()) : 0.0;
  out.complementarity = slack.size()
      ? sol.multipliers.cwiseProduct(slack).cwiseAbs().maxCoeff() : 0.0;
  return out;
}

}  // namespace qp
}  // namespace sightline
