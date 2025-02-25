from ultralytics import YOLO

class ModelLoader:  # Class and file should have the same name
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load_yolo_model(self):
        """
        Loads the YOLO model and returns both the model and class labels.
        """
        try:
            self.model = YOLO(self.model_path)
            self.model.eval()
            class_labels = self.model.names  # Get class labels from model
            return self.model, class_labels  # Return both
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")
