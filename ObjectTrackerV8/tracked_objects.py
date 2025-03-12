class TrackedObjects:
    """Represents a single tracked object."""
    def __init__(self, track_id, class_id, class_label, bounding_box):
        self.track_id = track_id
        self.class_id = class_id
        self.class_label = class_label
        self.bounding_box = bounding_box


    def __repr__(self):
        print(f"TrackedObject(ID={self.track_id}, Class={self.class_label}, BBox={self.bounding_box})")