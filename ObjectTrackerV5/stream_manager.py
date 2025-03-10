import cv2


class StreamManager:
    def __init__(self, input_media_source, process_frame):
        self.input_media_source = input_media_source
        self.process_frame = process_frame


    def process_video(self):
        """
        Process a video for object tracking.
        """
        video_capture = cv2.VideoCapture(self.input_media_source)
        if not video_capture.isOpened():
             raise ValueError(f"Error: Could not open {self.input_media_source}")

        while video_capture.isOpened():
            frame_available, frame = video_capture.read()
            if not frame_available:
                break
            tracked_objects = self.process_frame(frame)
            yield frame, tracked_objects  # Yield each frame with tracking results pass to line intrude later on
        video_capture.release()