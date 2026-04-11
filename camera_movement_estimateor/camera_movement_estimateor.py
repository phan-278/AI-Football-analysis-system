import pickle
import cv2
import numpy as np


class CameraMovementEstimator():
    def __init__(self,frame):
        
        first_frame_grayscle = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscle)
        mask_features[:,0:20]=1
        mask_features[:,900:10500]=1

        self.feature = print()

    def get_camera_movement(self,frames,read_from_stub = False,stub_path = None):
        # Read from stub

        camera_movement = [[0,0]*len(frames)]

        previous_frame = cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY)
        old_feature = cv2.goodFeaturesToTrack(previous_frame,)