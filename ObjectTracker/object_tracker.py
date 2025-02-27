import cv2
from ObjectTracker.model_loader import ModelLoader
from ObjectTracker.utils import get_class_ids_from_names
# from trackers.byte_tracker import BYTETracker  # Use YOLO's built-in ByteTrack


class ObjectTracker:
    def __init__(self, model_path, conf_threshold, objects_to_track, device = "cuda"):
        """
        Initialize the Object Tracker with YOLO and ByteTrack.
        """
        self.model, self.class_labels= ModelLoader(model_path).load_yolo_model()
        self.conf_threshold = conf_threshold
        self.expected_class_ids = get_class_ids_from_names(self.class_labels, objects_to_track)
        self.device = device
        # self.tracker = BYTETracker(track_thresh=0.5, match_thresh=0.8) # Initialize ByteTrack with tracking parameters


    def process_frame(self, frame):
        """
        Perform object detection & tracking using YOLO’s built-in tracker.
        """
        # Use YOLO's built-in ByteTrack tracking
        detection_results = self.model.track(frame, tracker="bytetrack.yaml", verbose=True)

        tracked_objects = []
        if detection_results and detection_results[0].boxes:  # Ensure boxes exist
            bounding_boxes = detection_results[0].boxes.xyxy.to(self.device)
            detected_class_ids = detection_results[0].boxes.cls.to(self.device)
            track_ids = detection_results[0].boxes.id.to(self.device)

            if detected_class_ids is None:
                detected_class_ids = []
            else:
                detected_class_ids = detected_class_ids.int().tolist()

            if track_ids is None:
                track_ids = [-1] * len(detected_class_ids)  # Assign -1 for missing track IDs
            else:
                track_ids = track_ids.int().tolist()

            for index, bbox in enumerate(bounding_boxes):
                class_id = detected_class_ids[index]
                track_id = track_ids[index]

                if class_id in self.expected_class_ids:
                    tracked_objects.append({
                        "track_id": track_id,
                        "class_id": class_id,
                        "class_label": self.class_labels[class_id],
                        "bounding_box": bbox.tolist()
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
            detected_objects = self.process_frame(frame)
            yield frame, detected_objects  # Yield each frame with tracking results
        video_capture.release()