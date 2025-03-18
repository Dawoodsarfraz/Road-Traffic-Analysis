    # Line intrusion detector use just line co-ordinates
    # use this class for line intrusion
    # for easy streaming manager which take streaming source. display stream function, take parameter do we want to display detection or not no need to
    # call streaming manager link of source
    # call stream video or show video implement detection true or false


import cv2
from ObjectTrackerV5.object_tracker import ObjectTracker
from ObjectTrackerV5.utils import annotate_frame  # Import annotation function
from ObjectTrackerV5.stream_manager import StreamManager
from ObjectTrackerV5.intrusion_detector import IntrusionDetector  # Import IntrusionDetector

def main():
    print("Starting Object Tracking...")

    model_path = "Models/Yolov12/weights/yolov12n.pt"
    # input_media_source = "./media/videos/222.mp4"
    input_media_source = 0  # For webcam
    # input_media_source = 'rtsp://getptz:a10alb8q9jz8jJiD@93.122.231.135:9554/ISAPI/Streaming/channels/102'

    objects_of_interest = ["person", "car"]
    conf_threshold = 0.3
    use_gpu = False

    tracker = ObjectTracker(model_path, conf_threshold, objects_of_interest, use_gpu)
    stream_manager = StreamManager(input_media_source, tracker.process_frame)

    # Pass True/False flag and the function to IntrusionDetector
    intrusion_detector = IntrusionDetector(use_annotate_frame=True, annotate_frame=annotate_frame)

    cv2.namedWindow("Object Tracking")
    cv2.setMouseCallback("Object Tracking", intrusion_detector.draw_polygon)

    for frame, tracked_objects in stream_manager.process_video():
        frame = intrusion_detector.detect_intrusion(frame, tracked_objects)
        cv2.imshow("Object Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
