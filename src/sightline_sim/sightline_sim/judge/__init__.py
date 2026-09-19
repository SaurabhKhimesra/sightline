"""Ground truth for the two rules. The planner never imports this.

R1 and R2 are defined in SPEC.md section 4. This package measures them from the
simulator's own state, which is exactly what the planner is not allowed to see.
"""

from .perception_truth import PerceptionTruth, match
from .rules import BlockedFrame, BlockEvent, Contact, R1Judge, R2Judge, SegmentationCamera

__all__ = ["PerceptionTruth", "match", "BlockedFrame", "BlockEvent", "Contact", "R1Judge", "R2Judge", "SegmentationCamera"]
