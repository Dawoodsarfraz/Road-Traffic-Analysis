import cv2
import sys
from ObjectTrackerV8.utils import annotate_frame
import ObjectTrackerV8.config as cfg
# from ObjectTrackerV8.stream_manager import

class LineIntrusionDetector:
    def __init__(self, display_annotated_frame):
        """
        :param display_annotated_frame: Boolean flag to enable or disable annotation
        """
        # Define fixed intrusion lines [(start_x, start_y), (end_x, end_y)]
        self.intrusion_lines = cfg.intrusion_lines
        self.prev_positions = {}
        self.display_annotated_frame = display_annotated_frame
        self.annotate_frame = annotate_frame
        self.frame_height = None
        self.frame_width = None


    def line_intersection(self, p1, p2, p3, p4):
        """ Checks if two line segments (p1-p2) and (p3-p4) intersect.
            p1 = prev_center
            p2 = bbox_center
            p3 = line_start
            p4 = line_end
        """
        def check_turn_orientation(point_A, point_B, point_C):
            return (point_C[1] - point_A[1]) * (point_B[0] - point_A[0]) > (point_B[1] - point_A[1]) * (point_C[0] - point_A[0])

        return check_turn_orientation(p1, p3, p4) != check_turn_orientation(p2, p3, p4) and check_turn_orientation(p1, p2, p3) != check_turn_orientation(p1, p2, p4)


    def is_crossing_line(self, obj_id, bbox):
        """
        Checks if an object crosses any defined line using its previous and current position.
        """
        if len(self.intrusion_lines) == 0:
            return False

        bbox_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2) # Compute object's center (current position)

        for line in self.intrusion_lines: # Check if intrusion lines are within image bounds
            for x, y in line:
                if not (0 <= x < self.frame_width and 0 <= y < self.frame_height):
                    print(f"Error: Line point {x, y} is outside the image bounds!")
                    sys.exit(1)

        prev_center = self.prev_positions.get(obj_id, bbox_center)  # Default to current position if unknown
        self.prev_positions[obj_id] = bbox_center  # Update position tracking

        for line_start, line_end in self.intrusion_lines:
            if self.line_intersection(prev_center, bbox_center, line_start, line_end):
                return True  # Intrusion detected
        return False


    def detect_intrusion(self, frame, tracked_objects):
        """
        Processes tracked objects, detects line intrusions, and applies annotation
        """

        self.frame_width, self.frame_height = frame.shape[:2]
        for line_start, line_end in self.intrusion_lines: # Draw defined lines on the frame
            cv2.line(frame, line_start, line_end, (0, 0, 255), 3)  # Red lines

        for obj in tracked_objects:
            bbox = list(map(int, obj.bounding_box))  # Convert to integers
            if len(bbox) != 4:
                continue  # Skip invalid bounding boxes

            intrusion = self.is_crossing_line(obj.track_id, bbox)
            color = (0, 0, 255) if intrusion else (0, 255, 0) # Draw bounding box: Red if intrusion, Green otherwise
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)

        if self.display_annotated_frame:
            frame = self.annotate_frame(frame, tracked_objects)

        return frame