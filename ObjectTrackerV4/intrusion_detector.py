import cv2
import numpy as np


class IntrusionDetector:
    def __init__(self, use_annotate_frame=True, annotate_frame=None):
        """
        :param use_annotate_frame: Boolean flag to enable or disable annotation
        :param annotate_frame: Function to annotate the frame
        """
        self.points = []
        self.zone_defined = False
        self.use_annotate_frame = use_annotate_frame
        self.annotate_frame = annotate_frame

    def draw_polygon(self, event, x, y, flags, param):
        """
        Handles mouse clicks to define the intrusion zone
        """
        if event == cv2.EVENT_LBUTTONDOWN:  # Left-click to add points
            self.points.append((x, y))
            # print(f"Point added: {x}, {y}")

        elif event == cv2.EVENT_RBUTTONDOWN and len(self.points) > 2:  # Right-click to finalize polygon
            self.zone_defined = True
            # print("Polygon defined! Right-click confirmed.")

    def is_inside_zone(self, bbox):
        """ Checks if a bounding box's center is inside the polygon """
        if not self.zone_defined or len(self.points) < 3:
            return False

        bbox_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

        # Ensure the polygon is in the correct shape
        polygon = np.array(self.points, np.int32).reshape((-1, 1, 2))

        result = cv2.pointPolygonTest(polygon, bbox_center, False)

        # print(f"BBox Center: {bbox_center}, Inside Zone: {result}")

        return result >= 0

    def detect_intrusion(self, frame, tracked_objects):
        """ Processes tracked objects, detects intrusions, and applies annotation """

        # Always draw the polygon-in-progress
        if len(self.points) > 1:
            cv2.polylines(frame, [np.array(self.points, np.int32)], isClosed=self.zone_defined, color=(0, 255, 255),
                          thickness=2)

        for obj in tracked_objects:
            bbox = obj.bounding_box  # Expecting [x1, y1, x2, y2]

            bbox = list(map(int, bbox))  # Convert bounding box values to integers

            if len(bbox) != 4:
                # print(f"Skipping invalid bbox: {bbox}")
                continue  # Skip invalid bounding boxes

            intrusion = self.is_inside_zone(bbox)
            # print(f"Intrusion Detected: {intrusion}, BBox: {bbox}")

            # Draw bounding box: Red if intrusion, Green otherwise
            color = (0, 0, 255) if intrusion else (0, 255, 0)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        if self.use_annotate_frame and callable(self.annotate_frame):  # Apply annotation function only if enabled
            frame = self.annotate_frame(frame, tracked_objects)

        return frame
