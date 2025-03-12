import cv2
from ObjectTrackerV7.object_tracker import ObjectTracker
from ObjectTrackerV7.utils import annotate_frame
from ObjectTrackerV7.zone_intrusion_detector import ZoneIntrusionDetector
import ObjectTrackerV7.config as cfg


class StreamManager:
    def __init__(self):
        """
        Initialize the StreamManager with all necessary components.
        """
        self.input_media_source = cfg.input_media_source


        # Initialize object tracker
        self.tracker = ObjectTracker(cfg.model_path,
                                     cfg.conf_threshold,
                                     cfg.objects_of_interest,
                                     cfg.use_gpu
                                     )

        # Initialize intrusion detector
        self.zone_intrusion_detector = ZoneIntrusionDetector(display_annotated_frame=True)


    def process_video(self):
        """
        Process a video for object tracking and intrusion detection.
        """
        video_capture = cv2.VideoCapture(self.input_media_source)
        if not video_capture.isOpened():
            raise ValueError(f"Error: Could not open {self.input_media_source}")

        cv2.namedWindow("Object Tracking")
        cv2.setMouseCallback("Object Tracking", self.zone_intrusion_detector.draw_polygon)

        while video_capture.isOpened():
            frame_available, frame = video_capture.read()
            if not frame_available:
                break

            # Perform object tracking
            tracked_objects = self.tracker.process_frame(frame)

            # Perform intrusion detection
            frame = self.zone_intrusion_detector.detect_intrusion(frame, tracked_objects)

            # Display frame
            cv2.imshow("Object Tracking", frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()