import cv2
import torch
from src.model_loader import ModelLoader
from experiment.utils import class_ids_from_names
from some_tracking_library import ByteTrack  # Replace with actual tracker import

class ObjectTracker:
    def __init__(self, model_path, conf_threshold, objects_to_track):
        """
        Initialize the Object Tracker.
        """
        self.model, self.class_labels, self.device = ModelLoader(model_path).load_yolo_model()
        self.conf_threshold = conf_threshold
        self.expected_class_ids = class_ids_from_names(self.class_labels, objects_to_track)
        self.tracker = ByteTrack()  # Initialize object tracker

    def process_frame(self, frame):
        """
        Perform object detection & tracking on a single frame.
        """
        if isinstance(frame, torch.Tensor):
            frame = frame.to(self.device)

        # Perform YOLO inference
        detection_results = self.model(frame, verbose=True)

        detected_objects = []
        if detection_results[0].boxes.data is not None:
            bounding_boxes = detection_results[0].boxes.xyxy.to(self.device)
            detected_class_ids = detection_results[0].boxes.cls.int().to(self.device).tolist()
            confidence_scores = detection_results[0].boxes.conf.to(self.device).tolist()

            detections = []
            for index, bbox in enumerate(bounding_boxes):
                class_id = detected_class_ids[index]
                confidence_score = confidence_scores[index]

                if class_id in self.expected_class_ids and confidence_score >= self.conf_threshold:
                    detections.append(bbox.tolist() + [confidence_score, class_id])

            # Perform tracking
            tracked_objects = self.tracker.update(detections)

            for track in tracked_objects:
                tracked_id = int(track[4])  # Track ID
                class_id = int(track[5])  # Object class
                bounding_box = track[:4]  # Bounding box coordinates

                detected_objects.append({
                    "track_id": tracked_id,
                    "class_id": class_id,
                    "class_label": self.class_labels[class_id],
                    "bounding_box": bounding_box
                })

        return detected_objects

    def process_image(self, input_media_source):
        """
        Process a single image for object tracking.
        """
        frame = cv2.imread(input_media_source)
        if frame is None:
            raise ValueError(f"Error: Could not open {input_media_source}")

        return self.process_frame(frame)

    def process_video(self, input_media_source):
        """
        Process a video for object tracking.
        """
        video_capture = cv2.VideoCapture(input_media_source)
        if not video_capture.isOpened():
            raise ValueError(f"Error: Could not open {input_media_source}")

        while video_capture.isOpened():
            frame_available, frame = video_capture.read()
            if not frame_available:
                break

            detected_objects = self.process_frame(frame)
            yield frame, detected_objects  # Yield each frame with tracking results

        video_capture.release()
