import cv2
from src.media_file_loader import LoadMediaFile
from src.model_loader import LoadModel

class ObjectDetection:
    def __init__(self, model, media_path, media_type, objects_name_list=None):
        """
        Initializes ObjectDetection with the model, media file, and media type (image or video).
        :param model: Pre-loaded model for detection.
        :param media_path: Path to the media file (image or video).
        :param media_type: Type of media (image or video).
        :param objects_name_list: List of object names to detect.
        """
        self.model = model
        self.classes_names = self.model.names
        self.media_path = media_path
        self.objects_name_list = objects_name_list or []  # expected objects # Detect all objects if no list is provided

        # Open the media (video or image) # remove media types just object detection in class
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
        :return: List of indices corresponding to the requested objects.
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
        Perform object detection on a single frame and save the result.
        :param frame: The frame to process.
        :param frame_idx: Frame index to name the output file (for videos).
        :return: Processed frame with bounding boxes.
        """
        object_indices = self.get_object_indices() # class ID's not indices
        if not object_indices: # make if condiction easy
            results = self.model.track(frame, imgsz = 640, verbose=True, persist=True) # track IDs
            # results = self.model(frame, verbose=False)  # Detect all objects
        else:
            results = self.model.track(frame, classes=object_indices, imgsz=640, verbose=True, persist = True) # Track iD's
            # results = self.model(frame, classes=object_indices, verbose=False)

        if results[0].boxes.data is not None:         # If detections exist, process the bounding boxes and labels
            boxes = results[0].boxes.xyxy.cpu()
            class_indices = results[0].boxes.cls.int().cpu().tolist()
            confidence_scores = results[0].boxes.conf.cpu()

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.int().tolist())
                class_idx = class_indices[i]
                confidence = confidence_scores[i].item()

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # make new function so we can use results to others
                label = f"{self.classes_names[class_idx]}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the processed frame using the save results
        self.save_results.save_frame(frame, frame_idx, self.media_type)
        return frame

    def detect_objects(self):
        """
        Main function to perform object detection on an image or video and save the results.
        """
        if self.media_type == 'video':
            frame_idx = 1
            while True:
                ret, frame = self.cap.read()  # Capture frame from video
                if not ret:
                    break

                # remove this and change name
                # Process and save the frame with detections
                processed_frame = self.process_frame(frame, frame_idx)
                cv2.imshow("Object Detection", processed_frame)

                # Exit if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q' or "Q"):
                    break

                frame_idx += 1

            # Release video capture object
            self.cap.release()
        elif self.media_type == 'image':
            # Process and save the image
            processed_image = self.process_frame(self.frame)
            cv2.imshow("Object Detection", processed_image)
            cv2.waitKey(0)

        cv2.destroyAllWindows()


def object_detection():
    """
    Initializes the ObjectDetection class and runs object detection.
    """
    # Load model and media path (which returns media type as well)
    model = load_model()
    media_path, media_type = get_media_file_path_and_type()  # Now returns path and type (image or video)



    # Get object names from the user
    # remove this function and
    objects_name = []
    while True:
        available_classes = list(ObjectDetection(model, media_path, media_type, []).classes_names.values())
        print(f"Available classes to detect: {', '.join(available_classes)}")

        object_name = input("Enter Name of Object You Want to Detect (or type 'quit' to exit, or leave empty for all objects): ").lower()
        if object_name == "quit":
            break
        elif object_name == "":
            break
        objects_name.append(object_name)

    # Initialize the ObjectDetection class with the requested objects
    detection_obj = ObjectDetection(model, media_path, media_type, objects_name)

    # Run object detection and save results
    detection_obj.detect_objects()
