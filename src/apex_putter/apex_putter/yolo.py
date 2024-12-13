"""
Use Yolo to identify golf balls.

Subscribers
-----------
    + /camera/camera/color/image_raw (sensor_msgs/msg/Image) - Subscribes to \
        the input image.

Publishers
----------
    + image_yolo (sensor_msgs/msg/Image) - Publishes the image with the \
        detections.
    + ball_detections (apex_putter_interfaces/msg/DetectionArray) - \
        Publishes an array of all detection center coordinates.

Parameters
----------
    + model (string) - The Yolo model to use: see docs.ultralytics.org for \
        available values. Default is best.pt.

"""

from apex_putter_interfaces.msg import Detection2D, DetectionArray
import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO


class YoloNode(Node):
    """
    Use Yolo to identify golf balls.

    Subscribes
    ----------
    image (sensor_msgs/msg/Image) - The input image

    Publishes
    ---------
    new_image (sensor_msgs/msg/Image) - The image with the detections
    detections (detection_msgs/msg/DetectionArray) - \
        Array of all detection center coordinates

    Parameters
    ----------
    model (string) - The Yolo model to use: see docs.ultralytics.org for \
        available values. Default is yolo11n.pt

    """

    def __init__(self):
        super().__init__('pose')
        self.bridge = CvBridge()
        self.declare_parameter('model',
                               value='best.pt')
        self.model = YOLO(self.get_parameter('model').
                          get_parameter_value().string_value)
        self.create_subscription(Image, '/camera/camera/color/image_raw',
                                 self.yolo_callback, 10)
        self.image_pub = self.create_publisher(Image, 'image_yolo', 10)
        self.detections_pub = self.create_publisher(DetectionArray,
                                                    'ball_detections', 10)

    def yolo_callback(self, image):
        """Identify all the objects in the scene."""
        # Convert to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
        # Run the model
        results = self.model(cv_image)

        # Create detecgtion message
        detections_msg = DetectionArray()

        # Process the results (draw bounding boxes on the image)
        for result in results:
            cv_image = result.plot()

            # Check if any of the detected objects is a person
            for box in result.boxes:
                x, y, w, h = box.xywh[0]  # center x, center y, width, height\
                self.get_logger().debug(
                    f'Detected object at ({x}, {y}) with \
                        width {w} and height {h}')
                center_x = int(x)
                center_y = int(y)

                # Create detection message and add to array
                detection = Detection2D()
                detection.x = center_x
                detection.y = center_y
                detections_msg.detections.append(detection)

                # Draw red dot (circle) at center
                # -1 fills the circle
                cv2.circle(cv_image, (center_x, center_y), 5, (0, 0, 255), -1)

        self.detections_pub.publish(detections_msg)

        new_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        # publish
        self.image_pub.publish(new_msg)


def main():
    rclpy.init()
    node = YoloNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
