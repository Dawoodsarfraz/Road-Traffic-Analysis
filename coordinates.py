import numpy as np

def xywh_to_xyxy(x, y, w, h):
    """
    Convert YOLO (x_center, y_center, width, height) format to (x_min, y_min, x_max, y_max).
    """
    x_min = x - w / 2
    y_min = y - h / 2
    x_max = x + w / 2
    y_max = y + h / 2
    return x_min, y_min, x_max, y_max


def xyxy_to_xywh(x_min, y_min, x_max, y_max):
    """
    Convert (x_min, y_min, x_max, y_max) to YOLO (x_center, y_center, width, height).
    """
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    return x_center, y_center, width, height


def normalize_coordinates(x_min, y_min, x_max, y_max, img_width, img_height):
    """
    Normalize bounding box coordinates between 0 and 1.
    """
    return (
        x_min / img_width,
        y_min / img_height,
        x_max / img_width,
        y_max / img_height
    )


def denormalize_coordinates(x_min, y_min, x_max, y_max, img_width, img_height):
    """
    Convert normalized coordinates back to pixel values.
    """
    return (
        int(x_min * img_width),
        int(y_min * img_height),
        int(x_max * img_width),
        int(y_max * img_height)
    )


def get_bbox_center(x_min, y_min, x_max, y_max):
    """
    Get the center point (x, y) of a bounding box.
    """
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    return x_center, y_center


def iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    box1 and box2 should be in (x_min, y_min, x_max, y_max) format.
    """
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    # Compute intersection
    x_min_inter = max(x_min1, x_min2)
    y_min_inter = max(y_min1, y_min2)
    x_max_inter = min(x_max1, x_max2)
    y_max_inter = min(y_max1, y_max2)

    inter_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)

    # Compute union
    box1_area = (x_max1 - x_min1) * (y_max1 - y_min1)
    box2_area = (x_max2 - x_min2) * (y_max2 - y_min2)

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0
