import cv2
from torch.xpu import device

from ObjectTracker.object_tracker import ObjectTracker  # Updated to use ObjectTracker
from ObjectTracker.utils import annotate_frame

def main():
    print("Starting Object Tracking...")

    model_path = "Models/Yolov12/weights/yolov12n.pt"
    input_media_source = "./media/videos/222.mp4"
    # input_media_source = "./media/images/image.png"  # For image tracking

    # Specify objects to track (None = track all objects)
    objects_to_track = ["car", "bus", "person"]
    conf_threshold = 0.5
    device = "cpu"
    tracker = ObjectTracker(model_path, conf_threshold, objects_to_track, device)

    # If the input is an image
    if input_media_source.lower().endswith(('.png', '.jpg', '.jpeg')):
        frame = cv2.imread(input_media_source)
        if frame is None:
            print(f"Error: Could not open {input_media_source}")
            return

        tracked_objects = tracker.process_image(input_media_source)
        annotated_frame = annotate_frame(frame, tracked_objects)

        cv2.imshow("Object Tracking", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # If the input is a video
    else:
        for frame, tracked_objects in tracker.process_video(input_media_source):
            annotated_frame = annotate_frame(frame, tracked_objects)
            cv2.imshow("Object Tracking", annotated_frame)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
