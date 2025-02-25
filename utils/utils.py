def get_target_class_ids(class_labels, target_object_classes=None):
    """
    Returns class IDs for the given target class names.
    If target_object_classes is None, it returns all available class IDs.
    """
    if not target_object_classes:  # Detect all classes if no specific target is given
        return list(class_labels.keys())

    return [class_id for class_id, class_name in class_labels.items() if class_name in target_object_classes]

