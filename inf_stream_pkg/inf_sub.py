import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import json

class YoloMultiCameraVisualizer(Node):
    def __init__(self):
        super().__init__('yolo_multi_visualizer')
        self.bridge = CvBridge()

        # Subscribe to three YOLO image topics
        self.create_subscription(CompressedImage, 'yolo1/image_annotated/compressed', self.image_callback1, 10)
        self.create_subscription(CompressedImage, 'yolo2/image_annotated/compressed', self.image_callback2, 10)
        self.create_subscription(CompressedImage, 'yolo3/image_annotated/compressed', self.image_callback3, 10)

        # Subscribe to three bounding box topics
        self.create_subscription(String, 'yolo1/bounding_boxes', self.bbox_callback1, 10)
        self.create_subscription(String, 'yolo2/bounding_boxes', self.bbox_callback2, 10)
        self.create_subscription(String, 'yolo3/bounding_boxes', self.bbox_callback3, 10)

        # Dictionary to store latest images (if any)
        self.image_dict = {
            "yolo1": None,
            "yolo2": None,
            "yolo3": None
        }

        self.get_logger().info("Multi-Camera YOLO Visualization Node Initialized")

        # Create a timer to refresh the display
        self.timer = self.create_timer(0.1, self.update_display)

    def image_callback1(self, msg):
        self.image_dict["yolo1"] = self.convert_image(msg)

    def image_callback2(self, msg):
        self.image_dict["yolo2"] = self.convert_image(msg)

    def image_callback3(self, msg):
        self.image_dict["yolo3"] = self.convert_image(msg)

    def bbox_callback1(self, msg):
        self.log_bboxes(msg, "Camera 1")

    def bbox_callback2(self, msg):
        self.log_bboxes(msg, "Camera 2")

    def bbox_callback3(self, msg):
        self.log_bboxes(msg, "Camera 3")

    def convert_image(self, msg):
        """Convert a ROS2 CompressedImage message to a CV2 image."""
        np_arr = np.frombuffer(msg.data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def log_bboxes(self, msg, camera_name):
        """Log bounding box messages for debugging."""
        try:
            bbox_data = json.loads(msg.data)
            self.get_logger().info(f"{camera_name} Bounding Boxes: {bbox_data}")
        except json.JSONDecodeError:
            self.get_logger().error(f"Failed to decode bounding box data from {camera_name}")

    def update_display(self):
        """Update the display with the latest images from all cameras."""
        width, height = 1280, 480  # Default image size
        blank_image = np.zeros((height, width, 3), dtype=np.uint8)  # Black image

        # Get latest images or use a blank image if none available
        img1 = self.image_dict["yolo1"] if self.image_dict["yolo1"] is not None else blank_image
        img2 = self.image_dict["yolo2"] if self.image_dict["yolo2"] is not None else blank_image
        img3 = self.image_dict["yolo3"] if self.image_dict["yolo3"] is not None else blank_image

        # Resize images to ensure consistency
        scale = 0.5
        width_re = int(width*scale)
        height_re = int(height*scale)
        img1 = cv2.resize(img1, (width_re, height_re))
        img1 = self.put_text(img1, 'infra-1')
        img2 = cv2.resize(img2, (width_re, height_re))
        img2 = self.put_text(img2, 'infra-2')
        img3 = cv2.resize(img3, (width_re, height_re))
        img3 = self.put_text(img3, 'infra-3')

        # Concatenate images horizontally
        combined_image = np.vstack((img1, img2, img3))

        # Display the combined image
        cv2.imshow("YOLO Multi-Camera View", combined_image)
        cv2.waitKey(1)
    
    def put_text(self, image, text):
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX

        # org
        org = (20, 20)

        # fontScale
        fontScale = 0.4
        
        # Blue color in BGR
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2
        
        # Using cv2.putText() method
        image = cv2.putText(image, text, org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        return image

def main(args=None):
    rclpy.init(args=args)
    node = YoloMultiCameraVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
