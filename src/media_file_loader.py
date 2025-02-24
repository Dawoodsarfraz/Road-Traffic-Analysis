import os
class LoadFile:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = None

    def load(self):
        try:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"File '{self.file_path}' not found.")
            self.file = self.file_path
            return self.file
        except Exception as e:
            raise ValueError(f"Failed to load file: {str(e)}")

# Usage Example:
file_path = "media/videos/222.mp4"  # Set your file path here
file_loader = LoadFile(file_path)
loaded_file = file_loader.load()
print(loaded_file)


