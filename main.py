from ObjectTrackerV7.config import display_annotated_frame
from ObjectTrackerV8.stream_manager import StreamManager

def main():
    print("Starting Object Tracking...")

    # input_media_source = "./media/videos/4.mp4"
    input_media_source = 0  # For webcam
    # # input_media_source = 'rtsp://getptz:a10alb8q9jz8jJiD@93.122.231.135:9554/ISAPI/Streaming/channels/102'


    # Initialize and start the stream manager
    display_annotated_frame = False
    stream_manager = StreamManager(input_media_source, display_annotated_frame)
    stream_manager.process_video()

    # stream_manager = StreamManager()
    # stream_manager.process_video()


if __name__ == "__main__":
    main()
