import os

class LoadMediaFile:
    def __init__(self, media_path):
        self.media_path = media_path

    def get_media_path(self):
        """Validate and return the media file path."""
        if not os.path.exists(self.media_path):
            raise FileNotFoundError(f"File '{self.media_path}' not found.")
        return self.media_path
