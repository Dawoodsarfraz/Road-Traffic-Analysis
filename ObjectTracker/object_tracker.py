import cv2
from ObjectTracker.model_loader import ModelLoader
from ObjectTracker.utils import get_class_ids_from_names


class ObjectTracker:
    def __init__(self, model_path, conf_threshold=0.5, objects_to_track=None, use_gpu=None):
        """
        Initialize the Object Tracker with YOLO and ByteTrack.
        """
        self.model, self.class_labels = ModelLoader(model_path).load_yolo_model()
        self.conf_threshold = conf_threshold
        self.expected_class_ids = get_class_ids_from_names(self.class_labels, objects_to_track)
        self.device = "cuda" if use_gpu == True else "cpu" # use Tru Falsa no input


    def process_frame(self, frame):
        """
        Perform object detection & tracking using YOLO’s built-in tracker while ensuring
        consistent object IDs across frames.
        """
        # Use YOLO's built-in ByteTrack tracking
        detection_results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=True)

        tracked_objects = []
        if detection_results and detection_results[0].boxes:
            conf_scores = detection_results[0].boxes.conf.to(self.device)
            valid_indices = conf_scores > self.conf_threshold  # picking those greater than confidence threshold
            bounding_boxes = detection_results[0].boxes.xyxy[valid_indices].to(self.device).tolist()
            detected_class_ids = detection_results[0].boxes.cls[valid_indices].to(self.device).tolist()
            track_ids = detection_results[0].boxes.id[valid_indices]

            # Ensure IDs are integers, otherwise assign -1
            track_ids = track_ids.int().tolist() if track_ids is not None else [-1] * len(detected_class_ids)

            for index, bbox in enumerate(bounding_boxes):
                class_id = int(detected_class_ids[index])
                track_id = int(track_ids[index])

                # Ensure only valid tracked objects are processed
                if track_id == -1:
                    continue  # Skipping untracked objects that have -1 Tracking ID

                if class_id in self.expected_class_ids: # create class for this and try frames having just entered objects not everything
                    tracked_objects.append({
                        "track_id": track_id,
                        "class_id": class_id,
                        "class_label": self.class_labels[class_id],
                        "bounding_box": bbox
                    })
        return tracked_objects


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