import cv2
from experiment.object_detector import ObjectDetector
from utils.utils import annotate_frame


def main():
    print("Starting Object Detection...")

    model_path = "Models/Yolov12/weights/yolov12n.pt"
    # input_source = "./media/videos/222.mp4"
    input_source = "./media/images/image.png"
    desired_objects = None # Track all objects
    conf_threshold = 0.5

    detector = ObjectDetector(model_path, conf_threshold, desired_objects)

    if input_source.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        frame = cv2.imread(input_source)
        if frame is None:
            print(f"Error: Could not open {input_source}")
            return

        detected_objects = detector.process_image(input_source)
        annotated_frame = annotate_frame(frame, detected_objects)

        cv2.imshow("Object Detection", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        for frame, detected_objects in detector.process_video(input_source):
            annotated_frame = annotate_frame(frame, detected_objects)  # Annotate in main.py
            cv2.imshow("Object Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
