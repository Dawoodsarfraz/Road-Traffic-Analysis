import cv2
from ZoneIntrusionDetector.object_tracker import ObjectTracker
from ZoneIntrusionDetector.utils import display_annotated_frame
from ZoneIntrusionDetector.zone_intrusion_detector import ZoneIntrusionDetector


class StreamManager:
    def __init__(self, input_media_source=None, zone_intrusion_points=None):
        """
        Initialize the StreamManager with all necessary components.
        """
        self.input_media_source = input_media_source
        self.model_path = "Models/Yolov12/weights/yolov12n.pt"
        self.objects_of_interest = ["person", "car"]
        self.conf_threshold = 0.3
        self.use_gpu = False
        self.zone_intrusion_points = zone_intrusion_points

        # Initialize object tracker
        self.tracker = ObjectTracker(
            self.model_path,
            self.conf_threshold,
            self.objects_of_interest,
            self.use_gpu
        )
        self.zone_intrusion_detector = ZoneIntrusionDetector(self.zone_intrusion_points) # Initialize intrusion detector with predefined points


    def process_video(self):
        """
        Process a video for object tracking and intrusion detection.
        """
        video_capture = cv2.VideoCapture(self.input_media_source)
        if not video_capture.isOpened():
            raise ValueError(f"Error: Could not open {self.input_media_source}")

        while video_capture.isOpened():
            frame_available, frame = video_capture.read()
            if not frame_available:
                break

            tracked_objects = self.tracker.process_frame(frame) # Perform object tracking
            self.zone_intrusion_detector.detect_intrusion(tracked_objects) # Perform intrusion detection
            frame = display_annotated_frame(frame, tracked_objects, self.zone_intrusion_points) # Draw annotations, including intrusion zone
            cv2.imshow("Object Tracking", frame) # Display frame

            if cv2.waitKey(1) & 0xFF == ord('q'): # Exit on 'q' key press
                break

        video_capture.release()
        cv2.destroyAllWindows()