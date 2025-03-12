import sys
from ObjectTrackerV8.model_loader import ModelLoader
from ObjectTrackerV8.utils import get_class_ids_from_names
from ObjectTrackerV8.tracked_objects import TrackedObjects


class ObjectTracker:
    def __init__(self, model_path, conf_threshold=0.5, objects_of_interest=None, use_gpu=False):
        """
        Initialize the Object Tracker .
        """
        self.model = ModelLoader(model_path, use_gpu).load_yolo_model()
        self.class_labels = self.model.names
        self.conf_threshold = conf_threshold
        self.expected_class_ids = get_class_ids_from_names(self.class_labels, objects_of_interest)
        self.device = "cuda" if use_gpu else "cpu"


    def process_tracked_objects(self, detection_results):
        """
        Process detection results and return a list of tracked objects.
        """
        tracked_objects = []
        print(f"Tracked Objects Memory Usage: {sys.getsizeof(tracked_objects)} bytes")

        if not detection_results or len(detection_results) == 0: # Check if detection_results exist and are not empty
            print("No detection results available!")
            return tracked_objects  # Return an empty list safely

        if detection_results[0].boxes is None or len(detection_results[0].boxes) == 0: # Check if any bounding boxes exist
            print("No objects detected in this frame!")
            return tracked_objects  # Return an empty list safely

        conf_scores = detection_results[0].boxes.conf.to(self.device)
        valid_indices = conf_scores > self.conf_threshold  # Filter confidence scores

        if valid_indices.sum() == 0:  # If no objects pass confidence threshold
            print("No valid detections (all below confidence threshold)!")
            return tracked_objects

        bounding_boxes = detection_results[0].boxes.xyxy[valid_indices].to(self.device).tolist()
        detected_class_ids = detection_results[0].boxes.cls[valid_indices].to(self.device).tolist()

        track_ids = detection_results[0].boxes.id
        if track_ids is None:
            track_ids = [-1] * len(detected_class_ids)  # Assign -1 if no tracking info
        else:
            track_ids = track_ids[valid_indices].int().tolist()


        for index, bbox in enumerate(bounding_boxes):
            class_id = int(detected_class_ids[index])
            track_id = int(track_ids[index])

            # Skip untracked objects
            if track_id == -1:
                continue

            # Append tracked object
            tracked_objects.append(TrackedObjects(
                track_id=track_id,
                class_id=class_id,
                class_label=self.class_labels[class_id],
                bounding_box=bbox
            ))

        return tracked_objects


    def process_frame(self, frame):

        detection_results = self.model.track(frame,
                                             persist=True,
                                             tracker="bytetrack.yaml",
                                             verbose=True,
                                             classes=self.expected_class_ids
                                             )

        return self.process_tracked_objects(detection_results)


    # def process_video(self, input_media_source):
    #     """
    #     Process a video for object tracking.
    #     """
    #     video_capture = cv2.VideoCapture(input_media_source)
    #     if not video_capture.isOpened():
    #         raise ValueError(f"Error: Could not open {input_media_source}")
    #
    #     while video_capture.isOpened():
    #         frame_available, frame = video_capture.read()
    #         if not frame_available:
    #             break
    #         tracked_objects = self.process_frame(frame)
    #         yield frame, tracked_objects  # Yield each frame with tracking results pass to line intrude later on
    #     video_capture.release()

        # class instead of video process