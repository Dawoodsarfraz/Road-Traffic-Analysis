import  cv2
import torch
from  src.model_loader import ModelLoader
from utils.utils import class_ids_from_names

class ObjectDetector:
    def __init__(self, model_path, conf_threshold, desired_objects):
        """
        Initialize the Object Detector.
        """
        self.model, self.class_labels, self.device = ModelLoader(model_path).load_yolo_model() # Get model, labels, and device
        self.conf_threshold = conf_threshold # Confidence threshold for filtering detections
        self.expected_class_ids = class_ids_from_names(self.class_labels, desired_objects) # Filter specific class IDs

    def process_frame(self, frame):
        """
        Perform object detection on a single frame.
        Optimized for both GPU and CPU execution.
        """
        # Move frame to the correct device (if it's a tensor)
        if isinstance(frame, torch.Tensor):
            frame = frame.to(self.device)

        # Perform inference
        detection_results = self.model(frame, verbose=True)

        detected_objects = []
        if detection_results[0].boxes.data is not None:
            # Keep bounding boxes, class IDs, and confidence scores on the same device
            bounding_boxes = detection_results[0].boxes.xyxy.to(self.device)
            detected_class_ids = detection_results[0].boxes.cls.int().to(self.device).tolist()
            confidence_scores = detection_results[0].boxes.conf.to(self.device).tolist()

            for index, bounding_box in enumerate(bounding_boxes):
                detected_class_id = detected_class_ids[index]
                confidence_score = confidence_scores[index]

                if detected_class_id in self.expected_class_ids and confidence_score >= self.conf_threshold:
                    detected_objects.append({
                        "class_id": detected_class_id,
                        "class_label": self.class_labels[detected_class_id],  # Add class label
                        "bounding_box": bounding_box.tolist(),  # Move to CPU when needed
                        "confidence": confidence_score
                    })

        return detected_objects

    def process_image(self, input_source):
        """
        Process a single image and return detection results.
        """
        frame = cv2.imread(input_source)
        if frame is None:
            raise ValueError(f"Error: Could not open {input_source}")

        return self.process_frame(frame)

    def process_video(self, input_source):
        """
        Process a video and return detection results for each frame.
        """
        video_capture = cv2.VideoCapture(input_source)
        if not video_capture.isOpened():
            raise ValueError(f"Error: Could not open {input_source}")

        results = []
        while video_capture.isOpened():
            frame_available, frame = video_capture.read()
            if not frame_available:
                break

            results.append(self.process_frame(frame))

        video_capture.release()
        return results
