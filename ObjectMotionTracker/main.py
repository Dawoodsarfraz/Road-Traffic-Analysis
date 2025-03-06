import cv2
from ObjectCounter.object_counter import ObjectCounter  # Updated to use ObjectTracker
from ObjectCounter.utils import annotate_frame


def main():
    print("Starting Object Tracking...")

    model_path = "../Models/Yolov12/weights/yolov12n.pt"
    input_media_source = "./media/videos/222.mp4"

    # Specify objects to track (None = track all objects)
    objects_of_interest = ["car", "bus", "person"]
    conf_threshold = 0.75
    use_gpu = False
    object_counter = ObjectCounter(model_path, conf_threshold, objects_of_interest, use_gpu)

    # Process video (passing file path directly)
    for frame, tracked_objects in object_counter.process_video(input_media_source):
        annotated_frame = annotate_frame(frame, tracked_objects)
        cv2.imshow("Object Tracking", annotated_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()