"""Real-time single-frame SMPL → G1 retargeter.

``SMPLRetargeter`` maps a (24, 7) SMPL body skeleton (per-joint position +
quaternion) to G1 29-DOF joint targets via mink differential IK. The robot
model (URDF/MJCF) is supplied by the caller as ``urdf_path`` — this package
carries no model assets of its own.
"""

from holosoma_inference.teleop.retargeting.realtime_smpl_retargeter import SMPLRetargeter

__all__ = ["SMPLRetargeter"]
