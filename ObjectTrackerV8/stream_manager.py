import cv2
from ObjectTrackerV8.object_tracker import ObjectTracker
from ObjectTrackerV8.line_intrusion_detector import LineIntrusionDetector
# import ObjectTrackerV8.config as cfg


class StreamManager:
    def __init__(self, input_media_source, display_annotated_frames):
        """
        Initialize the StreamManager with all necessary components.
        """
        self.input_media_source = input_media_source
        self.display_annotated_frames = display_annotated_frames
        self.model_path = "Models/Yolov12/weights/yolov12n.pt"
        self.objects_of_interest = ["person", "car", "cell phone"]
        self.conf_threshold = 0.3
        self.use_gpu = False
        self.tracker = ObjectTracker(self.model_path, self.conf_threshold, self.objects_of_interest, self.use_gpu) # Initialize object tracker
        self.line_intrusion_detector = LineIntrusionDetector(self.display_annotated_frames) # Initialize intrusion detector


    def process_video(self):
        """
        Process a video for object tracking and intrusion detection.
        """
        video_capture = cv2.VideoCapture(self.input_media_source)
        if not video_capture.isOpened():
            raise ValueError(f"Error: Could not open {self.input_media_source}")
        cv2.namedWindow("Object Tracking")
        while video_capture.isOpened():
            frame_available, frame = video_capture.read()
            if not frame_available:
                break

            tracked_objects = self.tracker.process_frame(frame) # Perform object tracking
            frame = self.line_intrusion_detector.detect_intrusion(frame, tracked_objects) # Perform intrusion detection
            cv2.imshow("Object Tracking", frame) # Display frame
            if cv2.waitKey(1) & 0xFF == ord('q'): # Exit on 'q' key press
                break

        video_capture.release()
        cv2.destroyAllWindows()