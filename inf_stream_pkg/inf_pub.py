# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO  # For YOLOv8
import os
from ament_index_python.packages import get_package_share_directory

class YoloPublisher(Node):
    def __init__(self):
        super().__init__('yolo_publisher')
        self.bridge = CvBridge()

        self.package_name = 'inf_stream_pkg'
        package_share_directory = get_package_share_directory(self.package_name)
        model_path = os.path.join(package_share_directory, 'best.pt')
        
        self.model = YOLO(model_path)  # Load YOLO model
        
        # Publishers
        self.image_pub = self.create_publisher(CompressedImage, 'yolo1/image_annotated/compressed', 10)
        self.bbox_pub = self.create_publisher(String, 'yolo1/bounding_boxes', 10)
        
        # Timer to simulate image capture and inference
        self.timer = self.create_timer(0.1, self.process_image)
        self.get_logger().info('YOLO Publisher Initialized')

    def process_image(self):
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(2)  # Keep camera open

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Failed to capture image")
            return

        # Resize image to optimize performance (assuming original input is correct)
        frame = cv2.resize(frame, (1280, 480))

        # Split the stereo image assuming left-right horizontal concatenation
        height, width, _ = frame.shape
        mid = width // 2
        left_image = frame[:, :mid, :]
        right_image = frame[:, mid:, :]

        # Process each image separately
        left_detections = self.run_yolo(left_image)
        right_detections = self.run_yolo(right_image)

        # Convert detections to a message
        bbox_msg = String()
        bbox_msg.data = str({
            "left": left_detections,
            "right": right_detections
        })
        self.bbox_pub.publish(bbox_msg)

        # Publish annotated images
        left_annotated = self.annotate_image(left_image, left_detections)
        right_annotated = self.annotate_image(right_image, right_detections)

        #left_img_msg = self.bridge.cv2_to_compressed_imgmsg(left_annotated)
        #right_img_msg = self.bridge.cv2_to_compressed_imgmsg(right_annotated)
        raw_img_msg = self.bridge.cv2_to_compressed_imgmsg(frame)
        self.image_pub.publish(raw_img_msg)
    '''
    def run_yolo(self, image):
        """Runs YOLO on the given image and returns bounding boxes."""
        results = self.model(image)
        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf.item()
                class_id = int(box.cls)
                label = f"{self.model.names[class_id]}: {confidence:.2f}"

                detections.append({
                    "label": self.model.names[class_id],
                    "confidence": confidence,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2
                })

        return detections
    '''
    def run_yolo(self, image):
        """Runs YOLO on the given image and returns bounding boxes and keypoints."""
        results = self.model(image)  # Run inference
        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf.item()
                class_id = int(box.cls)
                label = f"{self.model.names[class_id]}: {confidence:.2f}"

                # Extract keypoints if available
                keypoints = []
                if hasattr(result, "keypoints") and result.keypoints is not None:
                    keypoints = result.keypoints.xy.cpu().numpy().tolist()  # Convert keypoints to list
                
                detections.append({
                    "label": self.model.names[class_id],
                    "confidence": confidence,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "keypoints": keypoints  # Store keypoints
                })

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
    node = YoloPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()