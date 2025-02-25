import cv2

def annotate_frame(frame, detected_objects):
    """
    Draws bounding boxes and labels on the frame for detected objects.
    Uses the filtered detections from `process_frame`.
    """
    if frame is None:
        print("Error: Received None frame in annotate_frame")
        return None

    # print(f"Annotating frame with {len(detected_objects)} objects")  # Debugging

    for obj in detected_objects:
        x1, y1, x2, y2 = map(int, obj["bounding_box"])
        class_label = obj["class_label"]
        confidence = obj["confidence"]

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label with confidence score
        label = f"{class_label}: {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame




def class_ids_from_names(class_labels, desired_objects=None):
    """
    Returns class IDs for the given target class names.
    If target_object_classes is None, it returns all available class IDs.
    """
    if not desired_objects:  # Detect all classes if no specific target is given
        return list(class_labels.keys())

    return [class_id for class_id, class_name in class_labels.items() if class_name in desired_objects]

