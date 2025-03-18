# from ZoneIntrusionDetector.stream_manager import StreamManager
#
#
# def main():
#     print("Starting Object Tracking...")
#
#     input_media_source = "./media/videos/222.mp4"
#     # input_media_source = 0  # For webcam
#     # # input_media_source = 'rtsp://getptz:a10alb8q9jz8jJiD@93.122.231.135:9554/ISAPI/Streaming/channels/102'
#
#     zone_intrusion_points = [(100, 200), (400, 250), (350, 500), (120, 450)]  # Example predefined polygon
#
#     stream_manager = StreamManager(input_media_source, zone_intrusion_points)
#     stream_manager.process_video()
#
#
# if __name__ == "__main__":
#     main()


from LineIntrusionDetector.stream_manager import StreamManager


def main():
    print("Starting Object Tracking...")

    # input_media_source = "./media/videos/222.mp4"
    input_media_source = 0  # For webcam
    # # input_media_source = 'rtsp://getptz:a10alb8q9jz8jJiD@93.122.231.135:9554/ISAPI/Streaming/channels/102'

    starting_points_of_intrusion_line = [(50, 400)]   # List of starting points
    ending_points_of_intrusion_line = [(1300, 200)]   # List of corresponding end points

    stream_manager = StreamManager(input_media_source, starting_points_of_intrusion_line, ending_points_of_intrusion_line)
    stream_manager.process_video()


if __name__ == "__main__":
    main()