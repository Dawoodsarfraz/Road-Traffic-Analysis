import cv2
import numpy as np

def display_annotated_frame(frame, tracked_objects, intrusion_zone_points=None):
    """
    Draws:
    - Bounding boxes around detected objects
    - Labels with class name and ID
    - Intrusion zone boundary (if provided)
    """
    # Draw Intrusion Zone (if points are given)
    if intrusion_zone_points and len(intrusion_zone_points) >= 3:
        polygon = np.array(intrusion_zone_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [polygon], isClosed=True, color=(0, 255, 255), thickness=2)  # Yellow boundary

    # Draw bounding boxes on tracked objects
    for obj in tracked_objects:
        bbox = list(map(int, obj.bounding_box))
        if len(bbox) != 4:
            continue

        color = (0, 0, 255) if obj.intrusion_detected else (0, 255, 0)  # Red if intrusion, Green otherwise

        # Draw bounding box
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        # Display label with Track ID
        label = f"{obj.class_label} [ID: {obj.track_id}]"
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame


def get_class_ids_from_names(class_labels, objects_to_track=None):
    """
    Returns class IDs for the given target class names.
    If target_object_classes is None, it returns all available class IDs.
    """
    if not objects_to_track or len(objects_to_track)==0:  # Detect all classes if no specific target is given
        return list(class_labels.keys())
    return [class_id for class_id, class_name in class_labels.items() if class_name in objects_to_track]