"""
Vision module for Apex Putter.

This nodes is responsible for detecting the golf ball and publishes \
    transformation for robot base.

Subscribers
-----------
/camera/camera/aligned_depth_to_color/image_raw : sensor_msgs.msg.Image
    Subscribes to the depth image from the camera.
/camera/camera/color/camera_info : sensor_msgs.msg.CameraInfo
    Subscribes to the camera info for depth image processing.
/camera/camera/color/image_raw : sensor_msgs.msg.Image
    Subscribes to the color image from the camera.
/ball_detections : apex_putter_interfaces.msg.DetectionArray
    Subscribes to the ball detection messages.

Publishers
----------
/robot_base_frame : geometry_msgs.msg.TransformStamped
    Publishes the transform of the robot base frame.
/ball_marker : visualization_msgs.msg.Marker
    Publishes markers for detected balls in RViz.

"""

import apex_putter.transform_operations as transOps
from apex_putter_interfaces.msg import DetectionArray
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped

import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CameraInfo, Image
import tf2_ros
from visualization_msgs.msg import Marker


class Vision(Node):
    def __init__(self):
        """Initialize the Vision node."""
        super().__init__('vision')
        self.bridge = CvBridge()

        # RS parameters
        self.intrinsics = None
        self._depth_info_topic = '/camera/camera/color/camera_info'

        self._depth_image_topic = \
            '/camera/camera/aligned_depth_to_color/image_raw'
        self._colored_image_topic = '/camera/camera/color/image_raw'

        self._latest_color_img = None
        self._latest_depth_img = None

        # Image subscribers
        self.sub_depth = self.create_subscription(
            Image, self._depth_image_topic, self.image_depth_callback, 1
        )
        self.sub_depth_info = self.create_subscription(
            CameraInfo, self._depth_info_topic, self.image_depth_info_callback, 1
        )
        self.sub_color = self.create_subscription(
            Image, self._colored_image_topic, self.image_color_callback, 1
        )

        # Known transform from apriltag to robot base
        self.atag_to_rbf_matrix = np.array([
            [0, 1, 0, 0.180],
            [0, 0, 1, -0.125],
            [1, 0, 0, -0.075],
            [0, 0, 0, 1]
        ])

        self.ball_radius = 21  # mm

        self.atag_to_rbf_transform = \
            transOps.htm_to_transform(self.atag_to_rbf_matrix)

        # Calibrated transform
        # GOLDEN
        # self.atag_to_rbf_transform = Transform()
        self.atag_to_rbf_transform.translation.x = 0.180
        self.atag_to_rbf_transform.translation.y = -0.120
        self.atag_to_rbf_transform.translation.z = -0.075
        self.atag_to_rbf_transform.rotation.x = -0.49032395482128516
        self.atag_to_rbf_transform.rotation.y = -0.49049696252323216
        self.atag_to_rbf_transform.rotation.z = -0.5028613983824991
        self.atag_to_rbf_transform.rotation.w = 0.5158735921722442

        # TF listener
        self.tf_buffer = tf2_ros.buffer.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.rbf_publisher = self.create_publisher(
            TransformStamped,
            '/robot_base_frame',
            10
        )

        # Dynamic broadcaster for robot base frame
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Static broadcaster for robot base frame
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        self.ball_detection_sub = self.create_subscription(
            DetectionArray,
            '/ball_detections',
            self.ball_detection_callback,
            10
        )

        self.balls_detected_array = None  # 2d pixel location
        self.balls_in_camera_frame = None  # 3d camera frame location

        # 3d camera frame \location compensated
        self.compensated_balls_in_camera_frame = None

        self.timer = self.create_timer(0.001, self.timer_callback)

        marker_qos = rclpy.qos.QoSProfile(
            depth=10, durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE
        )
        self.ball_marker_publisher = self.create_publisher(
            Marker, 'ball_marker', marker_qos
        )

        self.get_logger().info('Vision node started.')

    def image_depth_callback(self, data):
        """Obtain the latest depth image."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            self._latest_depth_img = cv_image
        except CvBridgeError as e:
            self.get_logger().error(
                'CvBridgeError in imageDepthCallback: {}'.format(e))

        except ValueError as e:
            self.get_logger().error(
                'ValueError in imageDepthCallback: {}'.format(e))

            return

    def image_depth_info_callback(self, cameraInfo):
        """Obtain depth camera information."""
        try:
            if self.intrinsics:
                return
            self.intrinsics = rs.intrinsics()

            self.intrinsics.width = cameraInfo.width
            self.intrinsics.height = cameraInfo.height
            self.intrinsics.ppx = cameraInfo.k[2]
            self.intrinsics.ppy = cameraInfo.k[5]
            self.intrinsics.fx = cameraInfo.k[0]
            self.intrinsics.fy = cameraInfo.k[4]
            if cameraInfo.distortion_model == 'plumb_bob':
                self.intrinsics.model = rs.distortion.brown_conrady
            elif cameraInfo.distortion_model == 'equidistant':
                self.intrinsics.model = rs.distortion.kannala_brandt4
            self.intrinsics.coeffs = list(cameraInfo.d)

            # DEBUG
            self.scale_factor = self.intrinsics.width / 450
            self.get_logger().info(
                f'intrinsics_width: {self.intrinsics.width}, \
                    depth_y: {self.intrinsics.height}')
        except CvBridgeError as e:
            self.get_logger().error('CvBridgeError in image_depth_info_callback: {}'.format(e))
            return

    def imageColorCallback(self, data):
        """Call for the latest color image."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            self._latest_color_img = cv_image
            self._latest_color_img_ts = data.header.stamp
        except CvBridgeError as e:
            self.get_logger().error('CvBridgeError in image_color_callback: {}'.format(e))
            return
        except ValueError as e:
            self.get_logger().error('ValueError in image_color_callback: {}'.format(e))
            return

    def publish_rbf(self):
        """Publish robot base frame."""
        robot_base_transform = TransformStamped()
        robot_base_transform.header.stamp = self.get_clock().now().to_msg()
        robot_base_transform.header.frame_id = 'robot_base_tag'
        robot_base_transform.child_frame_id = 'robot_base_frame'
        robot_base_transform.transform = self.atag_to_rbf_transform
        self.rbf_publisher.publish(robot_base_transform)
        self.static_broadcaster.sendTransform(robot_base_transform)

        # Publish a uniform transform from RBF to real robot base
        rbf_to_base_transform = TransformStamped()
        rbf_to_base_transform.header.stamp = self.get_clock().now().to_msg()
        rbf_to_base_transform.header.frame_id = 'robot_base_frame'
        rbf_to_base_transform.child_frame_id = 'base'
        self.static_broadcaster.sendTransform(rbf_to_base_transform)

    def deproject_depth_point(self, x, y):
        """
        Convert pixel coordinates to real-world coordinates using depth info.

        Args:
        ----
            x (int): X-coordinate.
            y (int): Y-coordinate.

        Returns
        -------
            Tuple[float, float, float]: Real-world coordinates (x, y, z).

        """
        if (
            self.intrinsics and
            self._latest_depth_img is not None and
            self._latest_color_img is not None
        ):
            depth_x = int(x)
            depth_y = int(y)
            # self.get_logger().info(f"depth_x: {depth_x}, depth_y: {depth_y}")
            # depth = self._latest_depth_img[depth_x, depth_y]
            depth = self._latest_depth_img[depth_y, depth_x]
            result = rs.rs2_deproject_pixel_to_point(
                self.intrinsics, [x, y], depth)

            x_new, y_new, z_new = result[0], result[1], result[2]
            return x_new, y_new, z_new

    def ball_detection_callback(self, msg):
        """Handle ball detection callback."""
        # Initialize empty array with 2 columns for x,y coordinates
        balls_detected = np.empty((0, 2))

        for detection in msg.detections:  # Note: using msg.points
            self.get_logger().debug(
                f'Detected ball at ({detection.x}, {detection.y})')
            # Create a 1x2 array for the current detection

            ball_detected = np.array([[detection.x, detection.y]])
            balls_detected = np.vstack((balls_detected, ball_detected))

        self.balls_detected_array = balls_detected
        # self.get_logger().info(
        #     f"Ball detected at {self.balls_detected_array}")

        # Deproject into 3D
        balls_camera_frame = np.empty((0, 3))
        compensated_bcf = np.empty((0, 3))
        if self.balls_detected_array is not None:
            for i in self.balls_detected_array:
                i_x = i[0]
                i_y = i[1]
                x, y, z = self.deproject_depth_point(i_x, i_y)
                # deprojected to ball centre.
                comp_x, comp_y, comp_z = transOps.compensate_ball_radius(
                    dx=x, dy=y, dz=z, R=self.ball_radius)
                i_array = np.array([x, y, z])
                i_array = i_array/1000  # Convert to meters
                balls_camera_frame = np.vstack((balls_camera_frame, i_array))
                comp_i_array = np.array([comp_x, comp_y, comp_z])
                comp_i_array = comp_i_array/1000  # Convert to meters
                compensated_bcf = np.vstack((compensated_bcf, comp_i_array))
        self.balls_in_camera_frame = balls_camera_frame
        self.compensated_balls_in_camera_frame = compensated_bcf

    def create_ball_marker(self, x, y, z):
        """Create a ball marker for RViz."""
        marker = Marker()
        marker.header.frame_id = 'camera_color_optical_frame'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'ball'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.a = 0.7
        marker.color.r = 0.8
        marker.color.g = 0.0
        marker.color.b = 0.2
        return marker

    def drop_ball_marker(self):
        """Drop a marker at the detected ball location in RViz."""
        if self.balls_in_camera_frame is not None:
            for i in self.balls_in_camera_frame:
                # self.get_logger().info(
                #     f"Dropping marker at {i[0], i[1], i[2]}")
                x = i[0]
                y = i[1]
                z = i[2]
                marker = self.create_ball_marker(x, y, z)
                self.ball_marker_publisher.publish(marker)

    def publish_ball_transform(self):
        """Publish transform of the detected ball in the robot base frame."""
        if self.balls_in_camera_frame is not None:
            for i in self.balls_in_camera_frame:
                x = i[0]
                y = i[1]
                z = i[2]
                camera_to_ball_transform = TransformStamped()
                camera_to_ball_transform.header.stamp = \
                    self.get_clock().now().to_msg()
                camera_to_ball_transform.header.frame_id = \
                    'camera_color_optical_frame'
                camera_to_ball_transform.child_frame_id = 'ball'
                camera_to_ball_transform.transform.translation.x = x
                camera_to_ball_transform.transform.translation.y = y
                camera_to_ball_transform.transform.translation.z = z

                self.tf_broadcaster.sendTransform(camera_to_ball_transform)

        if self.compensated_balls_in_camera_frame is not None:
            for i in self.compensated_balls_in_camera_frame:
                x = i[0]
                y = i[1]
                z = i[2]
                camera_to_ball_transform = TransformStamped()
                camera_to_ball_transform.header.stamp = \
                    self.get_clock().now().to_msg()
                camera_to_ball_transform.header.frame_id = \
                    'camera_color_optical_frame'
                camera_to_ball_transform.child_frame_id = \
                    'ball_compensated'
                camera_to_ball_transform.transform.translation.x = x
                camera_to_ball_transform.transform.translation.y = y
                camera_to_ball_transform.transform.translation.z = z
                self.tf_broadcaster.sendTransform(camera_to_ball_transform)

    def publish_target_transform(self):
        """Offset the target marker by a distance."""
        # Look up tag15 transform
        try:
            tag15_transform = self.tf_buffer.lookup_transform(
                'camera_color_optical_frame', 'tag_15', rclpy.time.Time())

            tag15_dx = tag15_transform.transform.translation.x
            tag15_dy = tag15_transform.transform.translation.y
            tag15_dz = tag15_transform.transform.translation.z

            comp_dx, comp_dy, comp_dz = transOps.compensate_ball_radius(
                dx=tag15_dx, dy=tag15_dy, dz=tag15_dz, R=0.03)

            target_transform = TransformStamped()
            target_transform.header.stamp = self.get_clock().now().to_msg()
            target_transform.header.frame_id = 'camera_color_optical_frame'
            target_transform.child_frame_id = 'target'
            target_transform.transform.translation.x = comp_dx
            target_transform.transform.translation.y = comp_dy
            target_transform.transform.translation.z = comp_dz

            self.tf_broadcaster.sendTransform(target_transform)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().debug(f'Error looking up tag_15 transform: {e}')

    def timer_callback(self):
        """Timer callback to publish transforms and markers."""
        self.publish_rbf()
        self.drop_ball_marker()
        self.publish_ball_transform()
        self.publish_target_transform()


def main(args=None):
    """Entry point for the vision node."""
    rclpy.init(args=args)
    vision = Vision()
    rclpy.spin(vision)
    vision.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
