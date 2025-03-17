class LineIntrusionDetector:
    def __init__(self, starting_points_of_intrusion_line, ending_points_of_instrusionl_line):
        """
        Initializes the Line Intrusion Detector with predefined intrusion start and end points.
        """
        self.starting_points_of_intrusion_line = starting_points_of_intrusion_line # [(50, 400)]   # List of starting points
        self.ending_points_of_intrusion_line = ending_points_of_instrusionl_line # [(1300, 200)]   # List of corresponding end points
        self.obj_previous_positions = {}  # Store previous positions of tracked objects


    def detect_intrusion(self, tracked_objects):
        """Detects if any tracked object crosses an intrusion line and updates its status."""
        for obj in tracked_objects:
            bbox = list(map(int, obj.bounding_box))
            if len(bbox) != 4:
                continue  # Skip invalid bounding boxes

            obj_current_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            obj_track_id = obj.track_id  # Unique ID of the object

            obj_previous_center = self.obj_previous_positions.get(obj_track_id, obj_current_center)
            self.obj_previous_positions[obj_track_id] = obj_current_center  # Update previous position

            # Check if the object crosses any intrusion line
            intrusion_detected = any(
                self.intersection_of_line(obj_previous_center, obj_current_center, starting_point_of_line, ending_point_of_line)
                for starting_point_of_line, ending_point_of_line in zip(self.starting_points_of_intrusion_line, self.ending_points_of_intrusion_line)
            )

            obj.intrusion_detected = intrusion_detected  # Update object status


    def intersection_of_line(self, obj_previous_center, obj_current_center, starting_point_of_line, ending_point_of_line):
        """
        Checks if two line segments (point1-point2) and (point3-point4) intersect using vector cross products.
        """

        def is_counter_clockwise(point1, point2, point3):
            return (point3[1] - point1[1]) * (point2[0] - point1[0]) > (point2[1] - point1[1]) * (point3[0] - point1[0])

        return ((is_counter_clockwise(obj_previous_center, starting_point_of_line, ending_point_of_line) != is_counter_clockwise(obj_current_center, starting_point_of_line, ending_point_of_line))
                and (is_counter_clockwise(obj_previous_center, obj_current_center, starting_point_of_line) != is_counter_clockwise(obj_previous_center, obj_current_center, ending_point_of_line)))
