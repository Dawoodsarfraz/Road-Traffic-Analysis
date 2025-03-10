import cv2


class StreamManager:
    def __init__(self, input_media_source, tracker, intrusion_detector):
        self.input_media_source = input_media_source
        self.tracker = tracker
        self.intrusion_detector = intrusion_detector


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

            # Process frame for object tracking
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
