# config.py

# Choose the input source (webcam, video file, or RTSP stream)
# input_media_source = "./media/videos/222.mp4"
# input_media_source = 0  # Webcam
# input_media_source = "rtsp://getptz:a10alb8q9jz8jJiD@93.122.231.135:9554/ISAPI/Streaming/channels/102"

# Model configurations
# model_path = "Models/Yolov12/weights/yolov12n.pt"
# objects_of_interest = ["person", "car", "cell phone"]
# conf_threshold = 0.3
# use_gpu = False


intrusion_lines = [
            [(50, 350), (450, 450)],  # Line 1 (Horizontal)
            [(150, 100), (150, 400)],  # Line 2 (Vertical)
        ]