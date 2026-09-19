"""A small quadratic programme solver, for the motion layer's per cycle problem.

    minimise  0.5 x' G x + a' x      subject to   C' x >= b

G is symmetric positive definite, so the problem has one answer. The method is the
dual active set of Goldfarb and Idnani 1983: start from the unconstrained minimum,
which is dual feasible, and bring constraints in one at a time, never losing dual
feasibility. That suits this use: six variables, a few dozen constraints, and a new
problem every 10 ms, where an interior point solver would be slower and a penalty
method would let a hard constraint bend.

The factorisation is recomputed whenever the active set changes instead of being
updated with Givens rotations. On six variables a QR costs a few microseconds, and
the bookkeeping that the update needs is where that algorithm usually goes wrong.

Reference: D. Goldfarb and A. Idnani, "A numerically stable dual method for solving
strictly convex quadratic programs", Mathematical Programming 27 (1983) 1 to 33.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Solution:
    x: np.ndarray
    active: list          # which constraints are tight
    multipliers: np.ndarray
    iterations: int
    feasible: bool


def solve(G: np.ndarray, a: np.ndarray, C: np.ndarray, b: np.ndarray,
          max_iter: int = 200, tol: float = 1e-9) -> Solution:
    """Solve the QP. C holds one constraint per column: C[:, i] . x >= b[i]."""
    G = np.asarray(G, float)
    a = np.asarray(a, float).ravel()
    n = G.shape[0]
    if C is None or C.size == 0:
        C = np.zeros((n, 0))
        b = np.zeros(0)
    C = np.asarray(C, float).reshape(n, -1)
    b = np.asarray(b, float).ravel()
    m = C.shape[1]

    L = np.linalg.cholesky(G)                 # G = L L'
    J = np.linalg.solve(L.T, np.eye(n))       # J J' = G inverse
    x = -(J @ (J.T @ a))                      # unconstrained minimum: dual feasible
    active: list = []
    u = np.zeros(0)

    for iteration in range(1, max_iter + 1):
        slack = C.T @ x - b
        if m:
            slack[active] = np.maximum(slack[active], 0.0)   # tight by construction
        p = int(np.argmin(slack)) if m else -1
        if m == 0 or slack[p] >= -tol:
            full = np.zeros(m)
            if active:
                full[active] = u
            return Solution(x=x, active=list(active), multipliers=full,
                            iterations=iteration, feasible=True)

        n_p = C[:, p]
        u = np.append(u, 0.0)
        added = False
        for _ in range(max_iter):
            q = len(active)
            if q:
                B = J.T @ C[:, active]
                Q, R = np.linalg.qr(B, mode="complete")
                star = J @ Q
                d = star.T @ n_p
                r = np.linalg.solve(np.triu(R[:q, :q]), d[:q])
                z = star[:, q:] @ d[q:]
            else:
                r = np.zeros(0)
                z = J @ (J.T @ n_p)

            # how far the dual can go before an active constraint would turn negative
            t1, blocking = np.inf, -1
            for k in range(q):
                if r[k] > tol:
                    ratio = u[k] / r[k]
                    if ratio < t1:
                        t1, blocking = ratio, k

            zz = float(n_p @ z)
            t2 = -slack[p] / zz if zz > tol else np.inf
            t = min(t1, t2)
            if not np.isfinite(t):
                return Solution(x=x, active=list(active), multipliers=np.zeros(m),
                                iterations=iteration, feasible=False)

            if np.isfinite(t2):
                x = x + t * z
            if q:
                u[:q] = u[:q] - t * r
            u[-1] = u[-1] + t

            if t2 <= t1 + tol:                 # the new constraint is now tight
                active.append(p)
                added = True
                break
            # otherwise an old constraint has hit zero: drop it and try again
            keep = [i for i in range(q) if i != blocking]
            active = [active[i] for i in keep]
            u = np.append(u[keep], u[-1])
            slack = C.T @ x - b
        if not added:
            return Solution(x=x, active=list(active), multipliers=np.zeros(m),
                            iterations=iteration, feasible=False)

    full = np.zeros(m)
    if active:
        full[active] = u
    return Solution(x=x, active=list(active), multipliers=full, iterations=max_iter, feasible=True)


def kkt_error(G, a, C, b, sol: Solution) -> dict:
    """How far a solution is from the optimality conditions. For tests and logs."""
    G, a = np.asarray(G, float), np.asarray(a, float).ravel()
    C = np.asarray(C, float).reshape(G.shape[0], -1)
    b = np.asarray(b, float).ravel()
    slack = C.T @ sol.x - b
    return {
        "stationarity": float(np.max(np.abs(G @ sol.x + a - C @ sol.multipliers))) if C.size else
        float(np.max(np.abs(G @ sol.x + a))),
        "primal": float(max(0.0, -slack.min())) if slack.size else 0.0,
        "dual": float(max(0.0, -sol.multipliers.min())) if sol.multipliers.size else 0.0,
        "complementarity": float(np.max(np.abs(sol.multipliers * slack))) if slack.size else 0.0,
    }
