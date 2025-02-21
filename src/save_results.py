import os
import cv2

class SaveResults:
    def __init__(self, results_folder="../results"):
        """
        Initializes the SaveResults class to handle saving the results.
        
        :param results_folder: Folder to save the results.
        """
        self.results_folder = results_folder
        self.video_folder = os.path.join(results_folder, 'videos')
        self.images_folder = os.path.join(results_folder, 'images')

        # Create results folders if they don't exist
        os.makedirs(self.video_folder, exist_ok=True)
        os.makedirs(self.images_folder, exist_ok=True)

    def save_frame(self, frame, frame_idx=None, media_type='video'):
        """
        Save a processed frame to the results folder based on media type.
        
        :param frame: The frame to save.
        :param frame_idx: Frame index to name the output file (for video).
        :param media_type: Media type ('video' or 'image').
        """
        if media_type == 'video' and frame_idx is not None:
            # Save processed frame with zero-padded index
            video_output_path = os.path.join(self.video_folder, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(video_output_path, frame)

        elif media_type == 'image':
            # Save processed image in images folder
            image_output_path = os.path.join(self.images_folder, "processed_image.jpg")
            cv2.imwrite(image_output_path, frame)

    def save_video(self, output_path, frame_rate=30):
        """
        Convert saved frames into a single video file.
        
        :param output_path: Path to save the video.
        :param frame_rate: Frame rate for the output video.
        """
        frame_files = sorted(
            [f for f in os.listdir(self.video_folder) if f.endswith(".jpg")],
            key=lambda x: int(x.split("_")[-1].split(".")[0])  # Sort by frame index
        )

        if not frame_files:
            print("No frames found to create a video.")
            return

        # Read the first frame to get dimensions
        first_frame = cv2.imread(os.path.join(self.video_folder, frame_files[0]))
        height, width, _ = first_frame.shape

        # Define video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

        # Add each frame to the video
        for frame_file in frame_files:
            frame_path = os.path.join(self.video_folder, frame_file)
            frame = cv2.imread(frame_path)
            if frame is not None:
                out.write(frame)

        out.release()
        print(f"Video saved at {output_path}")
