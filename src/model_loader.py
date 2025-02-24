from ultralytics import YOLO

class LoadModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    def load_yolo_model(self):
        try:
            self.model = YOLO(self.model_path)
            self.model.eval()
            return self.model
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")

# Usage Example:
model_path = "utils/models/Yolov12/weights/yolov12n.pt"  # Set your model path here
model_loader = LoadModel(model_path)
model = model_loader.load_yolo_model()



