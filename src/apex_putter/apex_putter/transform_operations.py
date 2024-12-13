import math

from geometry_msgs.msg import Pose, Transform, TransformStamped

import numpy as np
from transforms3d.affines import compose, decompose
from transforms3d.quaternions import mat2quat, quat2mat


def htm_to_transform(htm: np.array) -> Transform:
    """
    Convert a homogeneous transformation matrix (HTM) to a ROS Transform object.

    Parameters
    ----------
    htm : np.array
        The 4x4 homogeneous transformation matrix.

    Returns
    -------
    Transform
        The corresponding ROS Transform object.

    """
    translation, rotation, _, _ = decompose(htm)
    quaternion = mat2quat(rotation)  # returns w,x,y,z

    result = Transform()
    result.translation.x = float(translation[0])
    result.translation.y = float(translation[1])
    result.translation.z = float(translation[2])
    result.rotation.w = float(quaternion[0])  # w
    result.rotation.x = float(quaternion[1])  # x
    result.rotation.y = float(quaternion[2])  # y
    result.rotation.z = float(quaternion[3])  # z

    return result


def transform_to_htm(transform: Transform) -> np.array:
    """
    Convert a ROS Transform object to a homogeneous transformation matrix (HTM).

    Parameters
    ----------
    transform : Transform
        The ROS Transform message.

    Returns
    -------
    np.array
        The corresponding 4x4 homogeneous transformation matrix.

    """
    translation = np.array([
        transform.translation.x,
        transform.translation.y,
        transform.translation.z
    ])
    quaternion = np.array([
        transform.rotation.w,
        transform.rotation.x,
        transform.rotation.y,
        transform.rotation.z
    ])
    rotation_matrix = quat2mat(quaternion)
    htm = np.eye(4)
    htm[0:3, 0:3] = rotation_matrix
    htm[0:3, 3] = translation
    return htm


def combine_transforms(known_matrix: np.array, tag_transform: TransformStamped) -> Transform:
    """
    Combine a known transform matrix with a TF2 transform in ROS2.

    Parameters
    ----------
    known_matrix : np.array
        A known 4x4 homogeneous transformation matrix.
    tag_transform : TransformStamped
        A transform from TF2.

    Returns
    -------
    Transform
        The resulting combined ROS Transform.

    """
    tf2_translation = np.array([
        tag_transform.transform.translation.x,
        tag_transform.transform.translation.y,
        tag_transform.transform.translation.z
    ])
    tf2_quaternion = [
        tag_transform.transform.rotation.w,
        tag_transform.transform.rotation.x,
        tag_transform.transform.rotation.y,
        tag_transform.transform.rotation.z
    ]
    tf2_rotation = quat2mat(tf2_quaternion)
    tf2_scale = np.ones(3)
    tf2_matrix = compose(tf2_translation, tf2_rotation, tf2_scale)

    # Combine transforms through matrix multiplication
    result_matrix = np.matmul(tf2_matrix, known_matrix)
    result = htm_to_transform(result_matrix)
    return result


def obj_in_bot_frame(T_camObj: np.array) -> np.array:
    """
    Convert a camera-to-object transform to an object-in-robot-frame transform.

    Parameters
    ----------
    T_camObj : np.array
        The 4x4 camera-to-object transform.

    Returns
    -------
    np.array
        The 4x4 object-to-robot-frame transform.

    """
    # TODO: Define the fixed frame transform T_botCam properly.
    # Placeholder: T_botCam currently not defined; must be replaced with correct transform.
    T_botCam = np.eye(4)
    T_objBot = np.dot(np.linalg.inv(T_camObj), np.linalg.inv(T_botCam))
    return T_objBot


def detected_obj_pose(T_camObj: Transform) -> Pose:
    """
    Compute the object's pose in the robot frame given the object's transform in the camera frame.

    Parameters
    ----------
    T_camObj : Transform
        The transform of the object in the camera frame.

    Returns
    -------
    Pose
        The pose of the object in the robot frame.

    """
    htm_camObj = transform_to_htm(T_camObj)
    T_objBot = obj_in_bot_frame(htm_camObj)

    pose = Pose()
    pose.position.x = T_objBot[0, 3]
    pose.position.y = T_objBot[1, 3]
    pose.position.z = T_objBot[2, 3]
    # Orientation calculation not implemented yet.
    return pose


def compensate_ball_radius(dx: float, dy: float, dz: float, R: float = 21) -> tuple:
    """
    Compensate for the ball radius when determining the ball center coordinates.

    Given the displacement from the camera to the ball (dx, dy, dz) and the ball radius R,
    this function returns the adjusted coordinates for the ball center.

    Parameters
    ----------
    dx : float
        Displacement in x.
    dy : float
        Displacement in y.
    dz : float
        Displacement in z.
    R : float, optional
        Radius of the ball, by default 21.

    Returns
    -------
    tuple
        (x_r, y_r, z_r) coordinates of the ball center.

    """
    distance = math.sqrt(dx**2 + dy**2 + dz**2)
    scaling_factor = (distance + R) / distance

    x_r = dx * scaling_factor
    y_r = dy * scaling_factor
    z_r = dz * scaling_factor

    return x_r, y_r, z_r


def test():
    """Test the transform conversion functions."""
    manipulator_pos = np.array([
        [0.7071, -0.7071, 0, 1],
        [0.7071, 0.7071, 0, 0.44454056],
        [0, 0, 1, 0.66401457],
        [0, 0, 0, 1]
    ])

    tranform = htm_to_transform(manipulator_pos)
    htm = transform_to_htm(tranform)
    print(htm)


if __name__ == '__main__':
    test()
