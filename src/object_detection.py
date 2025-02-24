import cv2
from src.model_loader import LoadModel
from src.media_file_loader import LoadMediaFile


class ObjectDetection:
    def __init__(self, target_classes, model_path, media_path):
        self.model = LoadModel(model_path).load_yolo_model()  # Pass model path

        # Create an instance of LoadMediaFile and call get_media_path()
        file_loader = LoadMediaFile(media_path)
        self.media_file = file_loader.get_media_path()  # Correct way

        self.class_names = self.model.names  # Get class names from model
        self.target_class_ids = self.get_target_class_ids(target_classes)  # Get class IDs of target objects

    def get_target_class_ids(self, target_classes):
        """Find class IDs for the given target class names."""
        return [idx for idx, name in self.class_names.items() if name in target_classes]

    def frame_process(self, frame):
        """Perform object detection."""
        results = self.model(frame, verbose=False)
        return results

    def draw_boxes(self, frame, results):
        """Draw bounding boxes only for the target objects."""
        if results[0].boxes.data is not None:
            boxes = results[0].boxes.xyxy.cpu()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            confidence_scores = results[0].boxes.conf.cpu().tolist()

            for index, box in enumerate(boxes):
                class_id = class_ids[index]

                if class_id in self.target_class_ids:  # Filter detections by target class IDs
                    x1, y1, x2, y2 = map(int, box.int().tolist())
                    confidence_score = confidence_scores[index]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{self.class_names[class_id]}: {confidence_score:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    def object_detection(self):
        """Reads video frames, processes them, and displays the output."""
        cap = cv2.VideoCapture(self.media_file)
        if not cap.isOpened():
            print(f"Error: Could not open {self.media_file}")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = self.frame_process(frame)  # Use frame_process() for detection + visualization
            processed_frame = self.draw_boxes(frame, processed_frame)  # Process and draw

            cv2.imshow("Object Detection", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
