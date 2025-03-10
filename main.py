    # Line intrusion detector use just line co-ordinates
    # use this class for line intrusion
    # for easy streaming manager which take streaming source. display stream function, take parameter do we want to display detection or not no need to
    # call streaming manager link of source
    # call stream video or show video implement detection true or false



import cv2
from ObjectTrackerV6.object_tracker import ObjectTracker
from ObjectTrackerV6.utils import annotate_frame
from ObjectTrackerV6.stream_manager import StreamManager
from ObjectTrackerV6.intrusion_detector import IntrusionDetector

def main():
    print("Starting Object Tracking...")

    model_path = "Models/Yolov12/weights/yolov12n.pt"
    input_media_source = 0  # Webcam (Change to a file path or RTSP URL)

    objects_of_interest = ["person", "car"]
    conf_threshold = 0.3
    use_gpu = False

    # Initialize tracker and intrusion detector
    tracker = ObjectTracker(model_path, conf_threshold, objects_of_interest, use_gpu)
    intrusion_detector = IntrusionDetector(use_annotate_frame=True, annotate_frame=annotate_frame)

    # Initialize and start the stream manager
    stream_manager = StreamManager(input_media_source, tracker, intrusion_detector)
    stream_manager.process_video()

if __name__ == "__main__":
    main()
