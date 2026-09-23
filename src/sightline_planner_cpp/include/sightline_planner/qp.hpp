// A small quadratic programme solver, for the motion layer's per cycle problem.
//
//     minimise  0.5 x' G x + a' x      subject to   C' x >= b
//
// G is symmetric positive definite, so the problem has one answer. The method is the
// dual active set of Goldfarb and Idnani: start from the unconstrained minimum, which
// is dual feasible, and bring constraints in one at a time. That suits six variables,
// a few dozen constraints and a new problem every 10 ms, where an interior point
// solver would be slower and a penalty method would let a hard constraint bend.
//
// D. Goldfarb and A. Idnani, "A numerically stable dual method for solving strictly
// convex quadratic programs", Mathematical Programming 27 (1983) 1 to 33.
#ifndef SIGHTLINE_PLANNER_QP_HPP
#define SIGHTLINE_PLANNER_QP_HPP

#include <vector>

#include "sightline_planner/types.hpp"

namespace sightline {
namespace qp {

struct Solution {
  VecX x;
  std::vector<int> active;   // which constraints are tight
  VecX multipliers;
  int iterations = 0;
  bool feasible = false;
};

struct KktError {
  double stationarity = 0.0;
  double primal = 0.0;
  double dual = 0.0;
  double complementarity = 0.0;
};

// Solve the QP. C holds one constraint per column: C.col(i) . x >= b[i].
Solution solve(const MatX &G, const VecX &a, const MatX &C, const VecX &b,
               int max_iter = 200, double tol = 1e-9);

// How far a solution is from the optimality conditions. For tests and logs.
KktError kkt_error(const MatX &G, const VecX &a, const MatX &C, const VecX &b,
                   const Solution &sol);

}  // namespace qp
}  // namespace sightline

#endif  // SIGHTLINE_PLANNER_QP_HPP
