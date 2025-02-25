import  cv2
from  src.model_loader import ModelLoader
from utils.utils import get_target_class_ids


class ObjectDetector:
    def __init__(self, model_file_path, conf_threshold, target_object_classes, media_source_path=None):
        """
        Initialize the Object Detector.
        """
        self.model, self.class_labels = ModelLoader(model_file_path).load_yolo_model() # return model and class labels
        self.conf_threshold = conf_threshold  # Confidence threshold for filtering detections
        self.target_class_ids = get_target_class_ids(self.class_labels, target_object_classes)  # Filter specific class IDs
        self.media_source = media_source_path  # Optional: Set media source path

    def process_frame(self, frame):
        """
        Perform object detection on a single frame.
        Returns filtered detection results.
        """
        detection_results = self.model(frame, verbose=False)

        filtered_detections = []
        if detection_results[0].boxes.data is not None:
            bounding_boxes = detection_results[0].boxes.xyxy.cpu()
            detected_class_ids = detection_results[0].boxes.cls.int().cpu().tolist()
            confidence_scores = detection_results[0].boxes.conf.cpu().tolist()

            for index, bounding_box in enumerate(bounding_boxes):
                detected_class_id = detected_class_ids[index]
                confidence_score = confidence_scores[index]

                if detected_class_id in self.target_class_ids and confidence_score >= self.conf_threshold:
                    filtered_detections.append({
                        "class_id": detected_class_id,
                        "bbox": bounding_box.tolist(),
                        "confidence": confidence_score
                    })
        return filtered_detections

    def process_image(self):
        """
        Process a single image and return detection results.
        """
        frame = cv2.imread(self.media_source)
        if frame is None:
            raise ValueError(f"Error: Could not open {self.media_source}")

        return self.process_frame(frame)

    def process_video(self):
        """
        Process a video and return detection results for each frame.
        """
        video_capture = cv2.VideoCapture(self.media_source)
        if not video_capture.isOpened():
            raise ValueError(f"Error: Could not open {self.media_source}")

        results = []
        while video_capture.isOpened():
            frame_available, frame = video_capture.read()
            if not frame_available:
                break

            results.append(self.process_frame(frame))

        video_capture.release()
        return results