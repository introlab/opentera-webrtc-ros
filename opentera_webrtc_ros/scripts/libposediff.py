# From https://linuxtut.com/en/80c8e0fa539ddff617e5/

# from tf.transformations import euler_from_quaternion, quaternion_from_euler
# from tf.transformations import compose_matrix, decompose_matrix, inverse_matrix
from geometry_msgs.msg import Pose, Point, Quaternion
import numpy as np
# from typing import List
# from numpy import ndarray

from pytransform3d.transformations import transform_from_pq, pq_from_transform, invert_transform


def transform_from_pose(pose: Pose) -> np.ndarray:
    """
    Convert a Pose to a 4x4 homogeneous transformation matrix.
    """
    return transform_from_pq((pose.position.x, pose.position.y, pose.position.z,
                             pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w))


def pose_from_transform(transform: np.ndarray) -> Pose:
    """
    Convert a 4x4 homogeneous transformation matrix to a Pose.
    """
    p, q = pq_from_transform(transform)
    return Pose(position=Point(*p), orientation=Quaternion(*q))

# def pose2homogeneousM(poseobj: Pose) -> ndarray:
#     # It says Quat to euler sxyz, but the order of XYZW is fine. Isn't it a little confusing?
#     tfeul = euler_from_quaternion(
#         [poseobj.orientation.x, poseobj.orientation.y, poseobj.orientation.z, poseobj.orientation.w], axes='sxyz')
#     # Description of translation amount
#     tftrans = [poseobj.position.x, poseobj.position.y, poseobj.position.z]
#     poseobjM = compose_matrix(angles=tfeul, translate=tftrans)
#     return poseobjM


# def homogeneousM2pose(Mat: ndarray):
#     _, _, angles, trans, _ = decompose_matrix(Mat)
#     quat = quaternion_from_euler(angles[0], angles[1], angles[2])
#     poseobj = Pose()
#     poseobj.orientation.x = quat[0]
#     poseobj.orientation.y = quat[1]
#     poseobj.orientation.z = quat[2]
#     poseobj.orientation.w = quat[3]
#     poseobj.position.x = trans[0]
#     poseobj.position.y = trans[1]
#     poseobj.position.z = trans[2]
#     return poseobj


# def pose_diff(p1, p2):
#     p1M = pose2homogeneousM(p1)
#     p2M = pose2homogeneousM(p2)
#     return homogeneousM2pose(p2M.dot(inverse_matrix(p1M)))
