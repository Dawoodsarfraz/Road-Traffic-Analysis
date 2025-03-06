import cv2
from ObjectTrackerV3.model_loader import ModelLoader
from ObjectTrackerV3.utils import get_class_ids_from_names
from ObjectTrackerV3.object_data import ObjectData
from ObjectTrackerV3.stream_manager import StreamManager

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
        if detection_results and detection_results[0].boxes:
            conf_scores = detection_results[0].boxes.conf.to(self.device)
            valid_indices = conf_scores > self.conf_threshold  # Filter confidence scores
            bounding_boxes = detection_results[0].boxes.xyxy[valid_indices].to(self.device).tolist()
            detected_class_ids = detection_results[0].boxes.cls[valid_indices].to(self.device).tolist()
            track_ids = detection_results[0].boxes.id[valid_indices]

            # Ensure IDs are integers, otherwise assign -1
            track_ids = track_ids.int().tolist() if track_ids is not None else [-1] * len(detected_class_ids)

            for index, bbox in enumerate(bounding_boxes):
                class_id = int(detected_class_ids[index])
                track_id = int(track_ids[index])

                # Skip untracked objects
                if track_id == -1:
                    continue

                    # Only process expected class IDs
                # if class_id in self.expected_class_ids: # can be removed
                tracked_objects.append(ObjectData(
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
    #         yield frame, tracked_objects  # Yield each frame with tracking results pass to line intude later on
    #     video_capture.release()

        # class instead of video process