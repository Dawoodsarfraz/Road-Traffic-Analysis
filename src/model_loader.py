import os
from ultralytics import YOLO
import re

class LoadModel:
    def __init__(self, model_folder: str):
        """
        Initialize the LoadModel class with the model folder path.
        :param model_folder: Path to the folder containing YOLO model files (.pt).
        """
        self.model_folder = model_folder
        self.older_version = []
        self.newer_version = []
        self.available_models = self.get_available_models()
        self.model = None

    def get_available_models(self) -> list:
        """
        Search for all available YOLO models in the given folder and categorize them.
        :return: A list of all model filenames found in the folder.
        """
        models = [f for f in os.listdir(self.model_folder) if f.endswith('.pt')]

        # Categorizing models dynamically
        for model in models:
            match = re.search(r"yolo(v?\d+)", model.lower())  # Extract version number
            if match:
                version_number = int(re.sub(r"\D", "", match.group(1)))  # Convert to integer
                if version_number <= 2:
                    self.older_version.append(model)  # Older YOLO models (v1-v4)
                elif version_number >= 3:
                    self.newer_version.append(model)  # Newer YOLO models (v5+)
        return models

    def load_yolo_model(self, model_name: str):
        """
        Load the selected YOLO model from the folder.
        :param model_name: Name of the YOLO model to load.
        :return: The loaded YOLO model.
        """
        model_path = os.path.join(self.model_folder, model_name)

        try:
            # Assign the categorized model lists for selection
            older_versions = self.older_version
            newer_versions = self.newer_version

            # Check model version for appropriate loading
            if model_name in newer_versions:
                self.model = YOLO(model_path)  # Load YOLOv5+ using ultralytics YOLO
                self.model.eval()  # Set to evaluation mode

            elif model_name in older_versions:
                
                raise ValueError(f"Custom loading required for older YOLO model: {model_name}")

            else:
                raise ValueError(f"Unsupported YOLO version or model: {model_name}")

            return self.model

        except Exception as e:
            raise ValueError(f"Failed to load model {model_name}: {str(e)}")


def load_model():
    """
    Loads a YOLO model from the predefined folder.
    :return: Loaded YOLO model or None if loading fails.
    """
    # Get the project root and model folder path
    current_script_path = os.path.abspath(__file__)
    parent_directory = os.path.dirname(current_script_path)
    project_root = os.path.dirname(parent_directory)

    model_folder = "models/Yolov11"  # Adjusted for relative path usage
    model_folder_path = os.path.join(project_root, model_folder)

    if not os.path.exists(model_folder_path):     # Check if the model folder exists
        print(f"Error: The folder '{model_folder}' does not exist at {model_folder_path}")
        return None

    model_loader = LoadModel(model_folder=model_folder_path)    # Initialize the LoadModel class and search for available models

    if not model_loader.available_models:
        print(f"No Models found in the folder '{model_folder_path}'.")
        return None

    print("\nAvailable models:")
    for idx, model in enumerate(model_loader.available_models, 1):
        print(f"{idx}. {model}")

    # User selects the model to load
    try:
        selected_model_index = int(input(f"\nEnter the number of the model you want to load (1-{len(model_loader.available_models)}): ")) - 1
        if not (0 <= selected_model_index < len(model_loader.available_models)):
            raise ValueError("Invalid model selection.")
        selected_model_name = model_loader.available_models[selected_model_index]
        return model_loader.load_yolo_model(selected_model_name)  # Return the loaded model
    except (ValueError, IndexError) as e:
        print(f"Invalid selection: {str(e)}")
        return None