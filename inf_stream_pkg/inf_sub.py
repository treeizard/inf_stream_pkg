import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from ultralytics import YOLO  # YOLOv8 Model
from ament_index_python.packages import get_package_share_directory

class YoloMultiCameraSubscriber(Node):
    def __init__(self):
        super().__init__('yolo_multi_camera_visualizer')
        self.bridge = CvBridge()

        # Load YOLO Model
        self.package_name = 'inf_stream_pkg'
        package_share_directory = get_package_share_directory(self.package_name)
        model_path = os.path.join(package_share_directory, 'best.pt')
        self.model = YOLO(model_path)  

        # Subscribe to three YOLO raw image topics
        self.create_subscription(CompressedImage, 'yolo1/image_raw/compressed', self.image_callback1, 10)
        self.create_subscription(CompressedImage, 'yolo2/image_raw/compressed', self.image_callback2, 10)
        self.create_subscription(CompressedImage, 'yolo3/image_raw/compressed', self.image_callback3, 10)

        # Dictionary to store latest images (if available)
        self.image_dict = {
            "yolo1": None,
            "yolo2": None,
            "yolo3": None
        }

        self.get_logger().info("YOLO Multi-Camera Inference Subscriber Initialized")

        # Timer to refresh and process images
        self.timer = self.create_timer(0.1, self.process_images)

    def image_callback1(self, msg):
        self.image_dict["yolo1"] = self.convert_image(msg)

    def image_callback2(self, msg):
        self.image_dict["yolo2"] = self.convert_image(msg)

    def image_callback3(self, msg):
        self.image_dict["yolo3"] = self.convert_image(msg)

    def convert_image(self, msg):
        """Convert a ROS2 CompressedImage message to a CV2 image."""
        np_arr = np.frombuffer(msg.data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def process_images(self):
        """Processes images from all feeds, splits them into left/right, runs YOLO in batch, and visualizes results."""

        width, height = 1280, 480  # Stereo image resolution
        blank_image = np.zeros((height, width, 3), dtype=np.uint8)  # Black image placeholder

        # Get images or use black images if not available
        img1 = self.image_dict["yolo1"] if self.image_dict["yolo1"] is not None else blank_image
        img2 = self.image_dict["yolo2"] if self.image_dict["yolo2"] is not None else blank_image
        img3 = self.image_dict["yolo3"] if self.image_dict["yolo3"] is not None else blank_image

        # Resize all images to ensure consistency
        img1 = cv2.resize(img1, (width, height))
        img2 = cv2.resize(img2, (width, height))
        img3 = cv2.resize(img3, (width, height))

        # Split stereo images into left and right images
        left1, right1 = self.split_stereo(img1)
        left2, right2 = self.split_stereo(img2)
        left3, right3 = self.split_stereo(img3)

        # Prepare batch for YOLO inference
        batch_images = [left1, right1, left2, right2, left3, right3]

        # Run YOLO model in batch
        detections = self.run_yolo(batch_images)

        # Annotate images with detections
        left1 = self.annotate_image(left1, detections[0])
        right1 = self.annotate_image(right1, detections[1])
        left2 = self.annotate_image(left2, detections[2])
        right2 = self.annotate_image(right2, detections[3])
        left3 = self.annotate_image(left3, detections[4])
        right3 = self.annotate_image(right3, detections[5])

        # Concatenate results for display (2 rows, 3 columns)
        top_row = np.hstack((left1, left2, left3))
        bottom_row = np.hstack((right1, right2, right3))
        combined_image = np.vstack((top_row, bottom_row))

        # Display the combined image
        cv2.imshow("YOLO Multi-Camera Inference", combined_image)
        cv2.waitKey(1)

    def split_stereo(self, image):
        """Splits a stereo image into left and right images."""
        height, width, _ = image.shape
        mid = width // 2
        left = image[:, :mid, :]
        right = image[:, mid:, :]
        return left, right

    def run_yolo(self, images):
        """Runs YOLO model on a batch of images."""
        results = self.model(images)
        detections = []

        for result in results:
            detection_list = []
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf.item()
                class_id = int(box.cls)

                detection_list.append({
                    "label": self.model.names[class_id],
                    "confidence": confidence,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2
                })

            detections.append(detection_list)

        return detections

    def annotate_image(self, image, detections):
        """Annotates the image with bounding boxes."""
        for det in detections:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            label = f"{det['label']}: {det['confidence']:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image

def main(args=None):
    rclpy.init(args=args)
    node = YoloMultiCameraSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
