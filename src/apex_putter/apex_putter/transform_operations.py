"""Functions to convert between different types of transforms."""

import math

from geometry_msgs.msg import Pose
from geometry_msgs.msg import Transform, TransformStamped

import numpy as np
from transforms3d.affines import compose, decompose
from transforms3d.quaternions import mat2quat, quat2mat


def htm_to_transform(htm: np.array) -> Transform:
    """
    Convert a HTM to a Transform object.

    Args
    ----
        htm format of the same (4x4 np.array): homogeneous transformation matrix.

    Returns
    -------
        transform: Transfrom is msg type of tf publisher.

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
    Convert a Transform object to a homogeneous transformation matrix.

    Args
    ----
        transform: Transfrom is msg type of tf publisher.

    Returns
    -------
        htm format of the same (4x4 np.array): \
            homogeneous transformation matrix

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


def combine_transforms(known_matrix: np.array,
                       tag_transform: TransformStamped) -> Transform:
    """
    Combine a known transform matrix with a TF2 transform in ROS2.

    Args
    ----
        known_transform (4x4 np.array): A known homogeneous \
            transformation matrix.
        tf2_transform (TransformStamped): Transform from TF2

    Returns
    -------
        Transform: The resulting combined transform

    """
    # Convert TF2 transform to matrix
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


def obj_in_bot_frame(T_camObj):
    """
    Return the pose of the object in the robot frame.

    Args
    ----
        T_camObj (4x4 np.array): Camera-to-Ball transform

    Returns
    -------
        4x4 np.array: T_objBot

    Fixed
    -----
        T_botCam

    """
    # Write the fixed frame transform here.
    T_botCam = np.array([0])
    T_objBot = np.dot(np.linalg.inv(T_camObj), np.linalg.inv(T_botCam))
    return T_objBot


def detected_obj_pose(T_camObj: Transform):
    """
    Return the pose of the detected object in the robot frame.

    This function returns the pose in robot frame of the robot to
    reach the object detected by vision.

    Args
    ----
        transform: tf of object detected in camera frame

    Returns
    -------
        pose: pose(or waypoint) of the object in robot frame.

    """
    T_camObj = transform_to_htm(T_camObj)
    T_objBot = obj_in_bot_frame(T_camObj)

    pose = Pose()
    pose.position.x = T_objBot[0, 3]
    pose.position.y = T_objBot[1, 3]
    pose.position.z = T_objBot[2, 3]
    # Orientation calculation not implemented yet.
    return pose


def compensate_ball_radius(dx, dy, dz, R=21):
    """
    Translate the pose from ball suface to ball's center.

    Args
    ----
        (x_c, y_c, z_c) : camera pose
        (x_b, y_b, z_b) : ball pose
        R: Radius of ball

    Returns
    -------
        (x_r, y_r, z_r) : pose for center of the ball.

    """
    # R: Radius of ball

    # Distance from the camera to the ball.
    distance = math.sqrt(dx**2 + dy**2 + dz**2)

    # Scale the displacement to account for the radius of the ball.
    scaling_factor = (distance + R) / distance

    # Coordinates of the ball center
    x_r = dx * scaling_factor
    y_r = dy * scaling_factor
    z_r = dz * scaling_factor

    return x_r, y_r, z_r


def test():
    """To test the above helper functions."""
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
