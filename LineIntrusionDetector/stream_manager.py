import cv2
from LineIntrusionDetector.object_tracker import ObjectTracker
from LineIntrusionDetector.line_intrusion_detector import LineIntrusionDetector
from LineIntrusionDetector.utils import display_annotated_frame


class StreamManager:
    def __init__(self, input_media_source=None, starting_points_of_intrusion_line=None, ending_points_of_intrusion_line=None):
        """
        Initialize the StreamManager with all necessary components.
        """
        self.input_media_source = input_media_source
        self.model_path = "Models/Yolov12/weights/yolov12n.pt"
        self.objects_of_interest = ["person", "car", "cell phone"]
        self.conf_threshold = 0.3
        self.use_gpu = False
        self.tracker = ObjectTracker(self.model_path, self.conf_threshold, self.objects_of_interest, self.use_gpu)

        # Store intrusion line points as instance variables
        self.starting_points_of_intrusion_line = starting_points_of_intrusion_line
        self.ending_points_of_intrusion_line = ending_points_of_intrusion_line

        # Initialize intrusion detector
        self.line_intrusion_detector = LineIntrusionDetector(
            self.starting_points_of_intrusion_line,
            self.ending_points_of_intrusion_line
        )

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

            # Check if frame is valid
            if not frame_available or frame is None:
                print("Warning: Empty frame received. Exiting loop.")
                break

            tracked_objects = self.tracker.process_frame(frame)  # Perform object tracking

            if not tracked_objects:
                print("Warning: No objects detected.")

            # Perform intrusion detection (updates object properties)
            self.line_intrusion_detector.detect_intrusion(tracked_objects)

            # Draw annotations (bounding boxes & intrusion lines)
            frame = display_annotated_frame(
                frame, tracked_objects,
                self.starting_points_of_intrusion_line,
                self.ending_points_of_intrusion_line
            )

            # Ensure frame is valid before displaying
            if frame is None or frame.size == 0:
                print("Warning: Processed frame is empty. Skipping display.")
                continue

            cv2.imshow("Object Tracking", frame)  # Display frame

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key press
                break

        video_capture.release()
        cv2.destroyAllWindows()