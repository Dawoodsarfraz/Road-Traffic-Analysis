import  cv2
from  src.model_loader import ModelLoader

class ObjectDetector:
    def __init__(self, model_file_path, conf_threshold, target_object_classes, media_source_path=None):
        """
        Initialize the Object Detector.
        """
        self.model = ModelLoader(model_file_path).load_yolo_model()  # Load YOLO model
        self.conf_threshold = conf_threshold  # Confidence threshold for filtering detections
        self.class_labels = self.model.names  # Get class names from model
        self.target_class_ids = get_target_class_ids(self.class_labels, target_object_classes)  # Filter specific class IDs
        self.media_source = media_source_path  # Optional: Set media source path
