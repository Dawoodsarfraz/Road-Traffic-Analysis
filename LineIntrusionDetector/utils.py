import cv2


def display_annotated_frame(frame, tracked_objects, starting_points_of_line=None, ending_points_of_line=None):
    """
    display_annotated_frame draws intrusion lines and bounding boxes, class label, Tracking ID on the frame.
    """

    # Draw intrusion lines (if provided)
    if starting_points_of_line and ending_points_of_line:
        for starting_point, ending_point in zip(starting_points_of_line, ending_points_of_line):
            cv2.line(frame, starting_point, ending_point, (0, 0, 255), 2)  # Red intrusion lines

    # Draw bounding boxes class labels and Tracking_ID
    for obj in tracked_objects:
        bbox = list(map(int, obj.bounding_box))
        class_label = obj.class_label
        track_id = obj.track_id

        if len(bbox) != 4:
            continue

        color = (0, 0, 255) if obj.intrusion_detected else (0, 255, 0)  # Red if intrusion, Green otherwise
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        # Display class label and track ID
        label = f"{class_label} [ID: {track_id}]"
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