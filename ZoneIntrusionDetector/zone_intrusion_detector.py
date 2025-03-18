import cv2
import sys
import numpy as np


class ZoneIntrusionDetector:
    def __init__(self, zone_intrusion_points):
        """
        :param zone_intrusion_points: List of (x, y) points defining the polygon.
        """
        self.zone_intrusion_points = zone_intrusion_points
        self.zone_defined = len(self.zone_intrusion_points) >= 3  # Only valid if it has 3+ points

    def is_inside_zone(self, bbox):
        """Checks if a bounding box's center is inside the polygon."""
        if not self.zone_defined:
            return False  # No valid zone defined yet

        bbox_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        polygon = np.array(self.zone_intrusion_points, np.int32).reshape((-1, 1, 2))
        return cv2.pointPolygonTest(polygon, bbox_center, False) >= 0

    def detect_intrusion(self, tracked_objects):
        """
        Updates each tracked object with intrusion status.
        """
        if not self.zone_defined:  # Exit early if not enough points
            print("Not enough points to define a valid Zone.")
            # sys.exit(1) # to exit full program
            return

        for obj in tracked_objects:
            obj.intrusion_detected = self.is_inside_zone(obj.bounding_box)  # Intrusion flag update