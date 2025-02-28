import cv2
from ObjectTracker.model_loader import ModelLoader
from ObjectTracker.utils import get_class_ids_from_names


class ObjectTracker:
    def __init__(self, model_path, conf_threshold, objects_to_track, device="cuda"):
        """
        Initialize the Object Tracker with YOLO and ByteTrack.
        """
        self.model, self.class_labels = ModelLoader(model_path).load_yolo_model()
        self.conf_threshold = conf_threshold
        self.expected_class_ids = get_class_ids_from_names(self.class_labels, objects_to_track)
        self.device = device
        self.object_tracker = {}
        self.tracked_ids = {}  # Dictionary to track assigned IDs

    def process_frame(self, frame):
        """
        Perform object detection & tracking using YOLO’s built-in tracker while ensuring
        consistent object IDs across frames.
        """
        # Use YOLO's built-in ByteTrack tracking
        detection_results = self.model.track(frame, tracker="bytetrack.yaml", verbose=True)

        tracked_objects = []
        if detection_results and detection_results[0].boxes:
            conf_scores = detection_results[0].boxes.conf.to(self.device)
            valid_indices = conf_scores > self.conf_threshold # picking those are greater to confidence threshold score
            bounding_boxes = detection_results[0].boxes.xyxy[valid_indices].to(self.device).tolist()
            detected_class_ids = detection_results[0].boxes.cls[valid_indices].to(self.device).tolist()
            track_ids = detection_results[0].boxes.id[valid_indices].to(self.device)

            track_ids = track_ids.int().tolist() if track_ids is not None else [-1] * len(detected_class_ids)

            for index, bbox in enumerate(bounding_boxes):
                class_id = int(detected_class_ids[index])
                track_id = int(track_ids[index])

                # Accessing bounding box values [x_min, y_min, x_max, y_max]
                center_of_x_axis, center_of_y_axis = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2  # Object center
                min_distance = float("inf")
                matched_id = track_id

                # Check if object was previously detected in previous frame
                for prev_id, (prev_center_of_x_axis, prev_center_of_y_axis) in self.object_tracker.items():
                    distance = ((center_of_x_axis - prev_center_of_x_axis) ** 2 + (center_of_y_axis - prev_center_of_y_axis) ** 2) ** 0.5
                    if distance < min_distance and distance < 50:  # Threshold to match same object
                        min_distance = distance
                        matched_id = prev_id

                # Update tracking dictionary
                self.object_tracker[matched_id] = (center_of_x_axis, center_of_y_axis)
                track_id = matched_id  # Assign consistent ID

                self.tracked_ids[track_id] = True  # Store assigned ID

                if class_id in self.expected_class_ids:
                    tracked_objects.append({
                        "track_id": track_id,
                        "class_id": class_id,
                        "class_label": self.class_labels[class_id],
                        "bounding_box": bbox
                    })

        return tracked_objects

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

            tracked_objects = self.process_frame(frame)
            yield frame, tracked_objects  # Yield each frame with tracking results

        video_capture.release()
