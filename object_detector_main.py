import cv2
from ObjectDetector.object_detector import ObjectDetector
from ObjectDetector.utils import annotate_frame


def main():
    print("Starting Object Detection...")

    model_path = "Models/Yolov8/weights/yolov8n.pt"
    input_media_source = "./media/videos/222.mp4"
    # media_source = "./media/images/image.png" # media_path

    # objects_to_detect = None  # Track all objects
    # objects_to_detect = [] # Track all objects
    objects_to_detect = None # ["car", "bus", "person", "wall", "moon", "sun", "kite"]
    conf_threshold = 0.5

    detector = ObjectDetector(model_path, conf_threshold, objects_to_detect)

    if input_media_source.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        frame = cv2.imread(input_media_source)
        if frame is None:
            print(f"Error: Could not open {input_media_source}")
            return

        detected_objects = detector.process_image(input_media_source)
        annotated_frame = annotate_frame(frame, detected_objects)

        cv2.imshow("Object Detection", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        for frame, detected_objects in detector.process_video(input_media_source):
            annotated_frame = annotate_frame(frame, detected_objects)  # Annotate in main.py
            cv2.imshow("Object Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q' or "Q"):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
