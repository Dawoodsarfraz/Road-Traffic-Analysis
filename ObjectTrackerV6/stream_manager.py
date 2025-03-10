import cv2
from ObjectTrackerV6.object_tracker import ObjectTracker
from ObjectTrackerV6.utils import annotate_frame
from ObjectTrackerV6.intrusion_detector import IntrusionDetector


class StreamManager:
    def __init__(self, input_media_source):
        """
        Initialize the StreamManager with all necessary components.
        """
        self.input_media_source = input_media_source

        # Define parameters (moved inside StreamManager)
        self.model_path = "Models/Yolov12/weights/yolov12n.pt"
        self.objects_of_interest = ["person", "car"]
        self.conf_threshold = 0.3
        self.use_gpu = False

        # Initialize object tracker
        self.tracker = ObjectTracker(self.model_path, self.conf_threshold, self.objects_of_interest, self.use_gpu)

        # Initialize intrusion detector
        self.intrusion_detector = IntrusionDetector(use_annotate_frame=True, annotate_frame=annotate_frame)

    def process_video(self):
        """
        Process a video for object tracking and intrusion detection.
        """
        video_capture = cv2.VideoCapture(self.input_media_source)
        if not video_capture.isOpened():
            raise ValueError(f"Error: Could not open {self.input_media_source}")

        cv2.namedWindow("Object Tracking")
        cv2.setMouseCallback("Object Tracking", self.intrusion_detector.draw_polygon)

        while video_capture.isOpened():
            frame_available, frame = video_capture.read()
            if not frame_available:
                break

            # Perform object tracking
            tracked_objects = self.tracker.process_frame(frame)

            # Perform intrusion detection
            frame = self.intrusion_detector.detect_intrusion(frame, tracked_objects)

            # Display frame
            cv2.imshow("Object Tracking", frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()