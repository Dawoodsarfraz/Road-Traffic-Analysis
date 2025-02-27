import cv2

def annotate_frame(frame, tracked_objects):
    """
    Draws bounding boxes, labels, and track IDs on the frame for tracked objects.
    Uses the filtered tracking data from `process_frame`.
    """
    if frame is None:
        print("Error: Received None frame in annotate_frame")
        return None

    for obj in tracked_objects:
        x1, y1, x2, y2 = map(int, obj["bounding_box"])
        class_label = obj["class_label"]
        track_id = obj["track_id"]  # Include tracking ID

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Draw bounding box
        label = f"{class_label} [Track ID: {track_id}]" # Display label with tracking ID
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame


def class_ids_from_names(class_labels, objects_to_track=None):
    """
    Returns class IDs for the given target class names.
    If target_object_classes is None, it returns all available class IDs.
    """
    if not objects_to_track or len(objects_to_track)==0:  # Detect all classes if no specific target is given
        return list(class_labels.keys())
    return [class_id for class_id, class_name in class_labels.items() if class_name in objects_to_track]

