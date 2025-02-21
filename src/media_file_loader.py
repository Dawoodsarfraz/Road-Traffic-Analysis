import os

class LoadMediaFile:
    def __init__(self, search_path="."):
        """
        Initializes the MediaFile class with the directory to search in.
        
        :param search_path: Directory to start the search (default is current directory).
        """
        self.search_path = os.path.abspath(search_path)  # Convert to absolute path
        self.available_files = self.get_available_files()

    def get_available_files(self) -> list:
        """
        Searches for all media files in the given directory.
        :return: List of all media files with their type (image or video).
        """
        media_extensions = {
            ".mp4": "video", ".avi": "video", ".mkv": "video", 
            ".jpg": "image", ".png": "image", ".jpeg": "image"
        }
        
        media_files = []
        for file in os.listdir(self.search_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in media_extensions:
                media_files.append((file, media_extensions[ext]))  # Store file along with its type
        return media_files

    def load_file(self, filename: str):
        """
        Searches for the file by its name and returns its full path if found.
        
        :param filename: The name of the file to search for.
        :return: Full file path if found.
        :raises: FileNotFoundError if the file is not found.
        """
        filename_lower = filename.lower()
        for file, _ in self.available_files:
            if file.lower() == filename_lower:
                return os.path.join(self.search_path, file)
        raise FileNotFoundError(f"File '{filename}' not found in '{self.search_path}'.")

    def get_media_type(self, filename: str) -> str:
        """
        Determines whether the file is an image or video based on its extension.
        
        :param filename: The name of the file.
        :return: 'image' or 'video' as a string.
        """
        # Pre-calculate the type during the initial scan of files
        filename_lower = filename.lower()
        for file, media_type in self.available_files:
            if file.lower() == filename_lower:
                return media_type
        return "unknown"  # Handle unknown types

def get_media_file_path_and_type():  # get media file paths 
    """
    Lists available media files and allows the user to select one.
    :return: Full path of the selected media file along with its type.
    """
    current_script_path = os.path.abspath(__file__)      # Determine project root
    parent_directory = os.path.dirname(current_script_path)  
    project_root = os.path.dirname(parent_directory)  

    media_folder = "media/videos"  # Define media folder path Adjust this if needed
    media_folder_path = os.path.join(project_root, media_folder)  # Use static paths, no user input

    if not os.path.exists(media_folder_path):     # Check if the media folder exists
        print(f"Error: The folder '{media_folder}' does not exist at {media_folder_path}")
        return None

    media_loader = LoadMediaFile(search_path=media_folder_path) # Initialize LoadMediaFile class and get available media files

    if not media_loader.available_files:
        print(f"No media files found in '{media_folder_path}'.")
        return None

    print("\nAvailable media files:")     # Display available files
    for idx, (file, media_type) in enumerate(media_loader.available_files, 1):
        print(f"{idx}. {file} ({media_type})")

    # Let the user select a file
    try:
        selected_index = int(input(f"\nEnter the number of the media file you want to select (1-{len(media_loader.available_files)}): ")) - 1
        if not (0 <= selected_index < len(media_loader.available_files)):
            raise ValueError("Invalid selection.")
        
        selected_file, file_type = media_loader.available_files[selected_index]
        file_path = media_loader.load_file(selected_file)
        print(file_path, file_type)

        return file_path, file_type  # Return the file path along with its type

    except (ValueError, IndexError) as e:
        print(f"Invalid selection: {str(e)}")
        return None
