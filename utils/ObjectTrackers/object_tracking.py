import cv2
from src.media_file_loader import get_media_file_path_and_type
from src.model_loader import load_model
from src.save_results import SaveResults


class ObjectTracking:
    def __init__(self, model, media_path, media_type, objects_name_list=None):
        """
        Initializes ObjectTracking with the model, media file, and media type (image or video).
        """
        self.model = model
        self.classes_names = self.model.names
        self.media_path = media_path
        self.media_type = media_type
        self.objects_name_list = objects_name_list or []  # Track all objects if no list is provided
        self.save_results = SaveResults()  # Initialize results manager

        if self.media_type == 'video':
            self.cap = cv2.VideoCapture(media_path)
            if not self.cap.isOpened():
                raise ValueError(f"Error: Could not open video file: {media_path}")
        elif self.media_type == 'image':
            self.frame = cv2.imread(media_path)
            if self.frame is None:
                raise ValueError(f"Error: Could not open image file: {media_path}")
        else:
            raise ValueError("Unsupported media type. Must be 'image' or 'video'.")

    def get_object_indices(self):
        """
        Returns the indices of the requested objects in the model's class names.
        """
        object_indices = []
        for obj in self.objects_name_list:
            if obj in self.classes_names.values():
                object_indices.append(list(self.classes_names.values()).index(obj))
            else:
                print(f"Object name {obj} not found in class list!")
        return object_indices

    def process_frame(self, frame, frame_idx=None):
        """
        Perform object tracking on a single frame and save the result.
        """
        object_indices = self.get_object_indices()
        if not object_indices:
            results = self.model.track(frame, imgsz=640, verbose=True, persist=True)  # Track all objects
        else:
            results = self.model.track(frame, classes=object_indices, imgsz=640, verbose=True,
                                       persist=True)  # Track selected objects

        if results[0].boxes.data is not None:  # If there are tracked objects
            boxes = results[0].boxes.xyxy.cpu()
            class_indices = results[0].boxes.cls.int().cpu().tolist()
            confidence_scores = results[0].boxes.conf.cpu()
            track_ids = results[0].boxes.id.cpu().tolist() if results[0].boxes.id is not None else []

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.int().tolist())
                class_idx = class_indices[i]
                confidence = confidence_scores[i].item()
                track_id = track_ids[i] if track_ids else "N/A"  # Get track ID

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{self.classes_names[class_idx]} [{track_id}]: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        self.save_results.save_frame(frame, frame_idx, self.media_type)
        return frame

    def track_objects(self):
        """
        Main function to perform object tracking on an image or video and save the results.
        """
        if self.media_type == 'video':
            frame_idx = 1
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                processed_frame = self.process_frame(frame, frame_idx)
                cv2.imshow("Object Tracking", processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                frame_idx += 1

            self.cap.release()
        elif self.media_type == 'image':
            processed_image = self.process_frame(self.frame)
            cv2.imshow("Object Tracking", processed_image)
            cv2.waitKey(0)

        cv2.destroyAllWindows()


def object_tracking():
    """
    Initializes the ObjectTracking class and runs object tracking.
    """
    model = load_model()
    media_path, media_type = get_media_file_path_and_type()
    objects_name = []
    tracking_obj = ObjectTracking(model, media_path, media_type, objects_name)
    tracking_obj.track_objects()
