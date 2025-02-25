import cv2
from src.object_detector import ObjectDetector
from utils.utils import annotate_frame

def main():
    print("Starting Object Detection...")

    # Define the model path and media file path
    model_path = "Models/Yolov12/weights/yolov12n.pt"  # Ensure the path is correct
    input_source = "./media/videos/222.mp4"  # Can be image or video
    # input_source = "./media/images/image.png"
    # Define target object classes and confidence threshold
    desired_objects = None # ["car", "bus", "person"]
    conf_threshold = 0.5

    # Initialize Object Detector
    detector = ObjectDetector(model_path, conf_threshold, desired_objects)

    def process_and_display(frame):
        detected_objects = detector.process_frame(frame)
        annotated_frame = annotate_frame(frame, detected_objects)
        cv2.imshow("Object Detection", annotated_frame)

    # Determine media type (image or video)
    if input_source.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        frame = cv2.imread(input_source)
        if frame is None:
            print(f"Error: Could not open {input_source}")
            return
        process_and_display(frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        video_capture = cv2.VideoCapture(input_source)
        if not video_capture.isOpened():
            print(f"Error: Could not open {input_source}")
            return

        while video_capture.isOpened():
            frame_available, frame = video_capture.read()
            if not frame_available:
                break
            process_and_display(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


