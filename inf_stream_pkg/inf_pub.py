import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np

class YoloPublisher(Node):
    def __init__(self):
        super().__init__('yolo_publisher')
        self.bridge = CvBridge()

        # Publisher for raw images
        self.image_pub = self.create_publisher(CompressedImage, 'yolo1/image_raw/compressed', 10)

        # Timer for capturing images
        self.timer = self.create_timer(0.1, self.process_image)
        self.get_logger().info('YOLO Image Publisher Initialized')

    def process_image(self):
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(2)  # Camera index

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Failed to capture image")
            return

        # Resize image (optional)
        frame = cv2.resize(frame, (1280, 480))

        # Convert and publish raw image
        raw_img_msg = self.bridge.cv2_to_compressed_imgmsg(frame)
        self.image_pub.publish(raw_img_msg)

def main(args=None):
    rclpy.init(args=args)
    node = YoloPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
