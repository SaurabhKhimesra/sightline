// The motion layer: every cycle, choose the joint velocity that does the job safely.
//
// The task wants a tool velocity. The rules want distances kept. Both go into one
// quadratic programme (docs/design.md section 8.5): the task sits in the objective, where it
// can be given up, and the rules sit in the constraints, where they cannot.
//
// Distances are held with the velocity damper of Faverjon and Tournassoud 1987: while
// a pair is further apart than the influence distance nothing binds, and inside it the
// closing speed is capped so that the pair can still be stopped before it reaches the
// safety distance. The person's own velocity is in the bound, so a hand moving toward
// the arm tightens it.
//
// R2 is the same damper against a line rather than a point: the sight line from the
// camera to a person voxel is a segment the arm must stay clear of, or it takes the
// camera's view of that voxel away.
#ifndef SIGHTLINE_PLANNER_MOTION_HPP
#define SIGHTLINE_PLANNER_MOTION_HPP

#include <limits>
#include <string>
#include <vector>

#include "sightline_planner/kin.hpp"
#include "sightline_planner/qp.hpp"
#include "sightline_planner/types.hpp"

namespace sightline {

// All scene choices, stated in docs/notes.md.
//
// A pair may close only as fast as it could still stop before its safety distance:
// approach speed <= sqrt(2 a (d - d_s)). The influence distance is where that allows
// the arm's top speed, so a link is never inside the zone faster than it can brake.
// The first version used the linear damper with a 10 cm zone for the worktop: a link
// arriving at 0.6 m/s needed 9 m/s^2 to stop, could not, and went 105 mm under it.
constexpr double kABrake = 3.0;        // m/s^2, what the planner counts on for braking a link (scene choice)
constexpr double kVLinkMax = 1.5;      // m/s, fastest a link point moves at the joint speed limit
constexpr double kBrakeZone = kVLinkMax * kVLinkMax / (2.0 * kABrake);   // 0.375 m
constexpr double kDSafe = 0.10;        // never closer than this to the person, m
constexpr double kDInfluence = kDSafe + kBrakeZone;
constexpr double kXi = 1.0;            // kept for reference: the linear damper this replaced
constexpr double kRVis = 0.06;         // keep this far off a sight line, m
constexpr double kVisInfluence = kRVis + kBrakeZone;
constexpr double kDSafeStatic = 0.02;  // the bench and the jig are not people
constexpr double kStaticInfluence = kDSafeStatic + kBrakeZone;
// A pair already inside the safety distance is not ordered to open it by the
// constraint: with several such pairs the orders conflict, every cycle is infeasible
// and the arm freezes where it is. The constraint only forbids getting closer, which
// standing still always satisfies against a still person, and backing out is asked
// for in the objective instead, with this weight per metre of shortfall.
constexpr double kRetreatWeight = 4.0;
constexpr int kPairsPerLink = 4;       // the nearest few obstacles per link is enough to shape the move
constexpr double kSmooth = 0.02;       // weight on small joint speeds
constexpr double kContinuity = 0.05;   // weight on staying near the last command

// How fast this pair may close: the speed from which the arm still stops in time.
double approach_limit(double dist, double safe);

// A segment of the cell the arm has to keep off, with the radius around it.
struct StaticSegment {
  Vec3 p0;
  Vec3 p1;
  double radius = 0.0;
};

// A plane the arm has to stay on the normal's side of.
struct Plane {
  Vec3 point;
  Vec3 normal;
};

// What the motion layer has to keep away from, in the world frame.
struct Obstacles {
  Points person = Points(0, 3);
  Points person_velocity = Points(0, 3);
  Points unseen = Points(0, 3);
  bool has_sight_from = false;
  Vec3 sight_from = Vec3::Zero();      // the eyes camera, for R2
  std::vector<StaticSegment> statics;
  std::vector<Plane> planes;

  // Person voxels and the space behind them: both can hold a hand.
  Points guard_points() const;
};

struct Report {
  Vec6 qd = Vec6::Zero();
  int constraints = 0;
  int r1_active = 0;
  int r2_active = 0;
  std::string fallback;
  double min_person_distance = std::numeric_limits<double>::infinity();  // to a voxel the camera actually saw
  double min_guard_distance = std::numeric_limits<double>::infinity();   // to anything guarded, unseen space included
  double solve_ms = 0.0;
};

// Closest point on segment ab to point p, and the distance.
void segment_distance(const Vec3 &p, const Vec3 &a, const Vec3 &b, Vec3 *closest, double *dist);

// Closest points between two segments, and the distance. Sampled at eight points, not
// solved: the capsules are short, the margins are centimetres, and an exact solve
// would cost more than the accuracy is worth at 100 Hz.
void segment_to_segment(const Vec3 &a0, const Vec3 &a1, const Vec3 &b0, const Vec3 &b1,
                        Vec3 *on_a, Vec3 *on_b, double *dist);

// Turns a wanted tool velocity into a joint velocity that keeps the rules.
class MotionLayer {
 public:
  explicit MotionLayer(ToolKinematics *kin, bool use_r1 = true, bool use_r2 = false,
                       double joint_speed = 90.0 * M_PI / 180.0,
                       double joint_accel = 400.0 * M_PI / 180.0, double dt = 0.01);

  // twist is the tool velocity the task wants: three linear, three angular.
  Report solve(const Vec6 &q, const Eigen::Matrix<double, 6, 1> &twist,
               const Obstacles &obstacles, const Vec6 *qd_prev = nullptr);

  // qd_task is the joint velocity the task wants. With nothing near it comes back as
  // it is, so a guarded controller and an unguarded one only differ near a person.
  Report solve_joint(const Vec6 &q, const Vec6 &qd_task, const Obstacles &obstacles,
                     const Vec6 *qd_prev = nullptr);

  const Vec6 &last() const { return last_; }
  void set_last(const Vec6 &qd) { last_ = qd; }

  // The rows the last solve built, for tests and logs.
  const std::vector<std::string> &labels() const { return labels_; }
  const qp::Solution &last_solution() const { return solution_; }

 private:
  struct Capsule {
    int geom = 0;
    int body = 0;
    double half = 0.0;
    double radius = 0.0;
  };
  struct WorldCapsule {
    int body = 0;
    Vec3 p0;
    Vec3 p1;
    double radius = 0.0;
  };
  struct NearPoint {
    int index = 0;
    Vec3 witness;
    Vec3 point;
    double dist = 0.0;
  };
  struct NearLine {
    Vec3 witness;
    Vec3 on_line;
    double dist = 0.0;
  };

  std::vector<Capsule> collision_shapes() const;
  std::vector<WorldCapsule> world_capsules(const Vec6 &q) const;
  Eigen::Matrix<double, 3, 6> point_jacobian(const Vec3 &point, int body) const;
  Report solve_core(const Vec6 &q, const Mat6 &G, Vec6 a, const Obstacles &obstacles,
                    const Vec6 &qd_prev);

  // The nearest few obstacle points to one capsule, with their witness points.
  std::vector<NearPoint> near_points(const Points &points, const Vec3 &p0, const Vec3 &p1,
                                     double radius, double influence) const;

  // Sight lines from the camera to person voxels that this capsule is close to. Lines
  // pointing nowhere near it, seen from the camera, are dropped before any distance is
  // computed: looping over every voxel for every link every cycle turned a 3.5 minute
  // episode into hours.
  std::vector<NearLine> near_lines(const Points &voxels, const Vec3 &eye, const Vec3 &p0,
                                   const Vec3 &p1, double radius, double influence) const;

  ToolKinematics *kin_;
  bool use_r1_;
  bool use_r2_;
  double joint_speed_;
  double joint_accel_;
  double dt_;
  std::vector<Capsule> capsules_;
  Vec6 last_ = Vec6::Zero();
  std::vector<std::string> labels_;
  qp::Solution solution_;
};

}  // namespace sightline

#endif  // SIGHTLINE_PLANNER_MOTION_HPP
