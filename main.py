from src.object_detection import ObjectDetection

def main():
    print("Starting Object Detection...")

    # Define the model path and media file path
    model_path = "Models/Yolov12/weights/yolov12n.pt"  # Ensure the path is correct
    media_source_path = "./media/videos/222.mp4"

    # Initialize Object Detection with model and media path
    target_object_classes = ["car", "bus", "person"]
    detector = ObjectDetection(target_object_classes, model_path, media_source_path, conf_threshold=0.5)  # Pass media_path
    detector.object_detection()

if __name__ == "__main__":
    main()
