import cv2
from src.model_loader import LoadModel
from src.media_file_loader import LoadMediaFile


class ObjectDetection:
    def __init__(self, target_object_classes, model_file_path, media_file_path):
        self.model = LoadModel(model_file_path).load_yolo_model()  # Load model with given path

        # Load media file path
        media_loader = LoadMediaFile(media_file_path)
        self.media_source = media_loader.get_media_path()

        self.class_labels = self.model.names  # Get class names from model
        self.target_class_ids = self.get_target_class_ids(target_object_classes)  # Get target class IDs

    def get_target_class_ids(self, target_object_classes):
        """Find class IDs for the given target class names."""
        return [class_id for class_id, class_name in self.class_labels.items() if class_name in target_object_classes]

    def process_frame(self, frame):
        """Perform object detection on a frame."""
        detection_results = self.model(frame, verbose=False)
        return detection_results

    def annotate_frame(self, frame, detection_results):
        """Draw bounding boxes only for the target objects."""
        if detection_results[0].boxes.data is not None:
            bounding_boxes = detection_results[0].boxes.xyxy.cpu()
            detected_class_ids = detection_results[0].boxes.cls.int().cpu().tolist()
            confidence_scores = detection_results[0].boxes.conf.cpu().tolist()

            for index, bounding_box in enumerate(bounding_boxes):
                detected_class_id = detected_class_ids[index]

                if detected_class_id in self.target_class_ids:  # Filter detections by target class IDs
                    x1, y1, x2, y2 = map(int, bounding_box.int().tolist())
                    confidence_score = confidence_scores[index]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{self.class_labels[detected_class_id]}: {confidence_score:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    def object_detection(self):
        """Process media file (image or video) and display results."""
        if self.media_source.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
            frame = cv2.imread(self.media_source)
            if frame is None:
                print(f"Error: Could not open {self.media_source}")
                return

            detection_results = self.process_frame(frame)  # Perform detection
            annotated_frame = self.annotate_frame(frame, detection_results)  # Draw bounding boxes

            cv2.imshow("Object Detection", annotated_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            video_capture = cv2.VideoCapture(self.media_source)
            if not video_capture.isOpened():
                print(f"Error: Could not open {self.media_source}")
                return

            while video_capture.isOpened():
                frame_available, frame = video_capture.read()
                if not frame_available:
                    break

                detection_results = self.process_frame(frame)  # Perform detection
                annotated_frame = self.annotate_frame(frame, detection_results)  # Draw bounding boxes

                cv2.imshow("Object Detection", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            video_capture.release()
            cv2.destroyAllWindows()
