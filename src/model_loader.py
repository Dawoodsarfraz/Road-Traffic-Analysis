import torch
from ultralytics import YOLO


class ModelLoader:
    """
    Loads the YOLO model and assigns it to the specified device (CPU/GPU).
    """
    def __init__(self, model_path, device=None):
        """
        Initializes the ModelLoader.

        :param model_path: Path to the YOLO model file.
        :param device: Device to load the model on ("cuda" or "cpu").
                       If None, it will use CUDA if available, otherwise CPU.
        """
        self.model_path = model_path
        self.model = None
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

    def load_yolo_model(self):
        """
        Loads the YOLO model, sets it to evaluation mode, moves it to the selected device,
        and returns the model, class labels, and device.

        :return: Tuple having (model, class_labels, device)
        """
        try:
            self.model = YOLO(self.model_path).to(self.device)  # Move model to device
            self.model.eval()  # Set model to evaluation mode
            self.class_labels = self.model.names  # Get class labels from model
            return self.model, self.class_labels, self.device
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")

