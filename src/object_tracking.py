import cv2
from model_loader import load_model
from media_file_loader import get_media_file_path_and_type

class ObjectTracking:
    def __init__(self):
        pass





def object_tracking():
    model = load_model() 
    media_path, media_type =  get_media_file_path_and_type()
    print("Hello World!!!")

if __name__=="__main__":
    object_tracking()