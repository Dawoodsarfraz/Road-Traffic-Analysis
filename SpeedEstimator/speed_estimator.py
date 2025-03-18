import time
import numpy as np


class SpeedEstimator:
    def __init__(self, frames_per_second=30, pixels_per_meter=10):
        """
        Initializes the Speed Estimator with predefined points for tracking speed.
        :param frames_per_second: Frames per second of the video.
        :param pixels_per_meter: Conversion factor for real-world speed calculation.
        """
        self.obj_previous_positions = {}  # Stores previous positions of objects
        self.obj_timestamps = {}  # Stores timestamps of previous positions
        self.frames_per_second = frames_per_second  # Frames per second (adjust based on actual video)
        self.pixels_per_meter = pixels_per_meter  # Pixel-to-meter conversion


    def estimate_speed(self, tracked_objects):
        """Calculates speed for each tracked object."""
        for obj in tracked_objects:
            bbox = list(map(int, obj.bounding_box))
            if len(bbox) != 4:
                continue  # Skip invalid bounding boxes

            obj_current_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            obj_track_id = obj.track_id  # Unique ID for each tracked object

            # Get previous position & time
            obj_previous_center = self.obj_previous_positions.get(obj_track_id, obj_current_center)
            previous_time = self.obj_timestamps.get(obj_track_id, time.time())

            # Update tracking info
            self.obj_previous_positions[obj_track_id] = obj_current_center
            self.obj_timestamps[obj_track_id] = time.time()

            # Calculate distance traveled in pixels
            distance_pixels = np.linalg.norm(np.array(obj_current_center) - np.array(obj_previous_center))

            # Calculate time difference
            time_diff = time.time() - previous_time  # Time in seconds
            if time_diff == 0:
                continue  # Avoid division by zero

            # Convert pixels to real-world distance (meters)
            distance_meters = distance_pixels / self.pixels_per_meter

            # Compute speed (meters per second)
            speed_mps = distance_meters / time_diff
            speed_kph = speed_mps * 3.6  # Convert to km/h

            # Store speed in the object
            obj.speed = round(speed_kph, 2)

            print(f"Object {obj.track_id} Speed: {obj.speed} km/h")  # Debugging output
