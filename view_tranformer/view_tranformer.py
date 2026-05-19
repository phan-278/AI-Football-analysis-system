import numpy as np 
import cv2
import os
import pickle
from ultralytics import YOLO

class SoccerPitchConfiguration:
    def __init__(self):
        self.width = 70.0  # standard width in meters
        self.length = 120.0  # standard length in meters
        self.penalty_box_width = 41.0
        self.penalty_box_length = 20.15
        self.goal_box_width = 18.32
        self.goal_box_length = 5.50
        self.centre_circle_radius = 9.15
        self.penalty_spot_distance = 11.00
        
        self.vertices = [
            (0, 0),  # 1
            (0, (self.width - self.penalty_box_width) / 2),  # 2
            (0, (self.width - self.goal_box_width) / 2),  # 3
            (0, (self.width + self.goal_box_width) / 2),  # 4
            (0, (self.width + self.penalty_box_width) / 2),  # 5
            (0, self.width),  # 6
            (self.goal_box_length, (self.width - self.goal_box_width) / 2),  # 7
            (self.goal_box_length, (self.width + self.goal_box_width) / 2),  # 8
            (self.penalty_spot_distance, self.width / 2),  # 9
            (self.penalty_box_length, (self.width - self.penalty_box_width) / 2),  # 10
            (self.penalty_box_length, (self.width - self.goal_box_width) / 2),  # 11
            (self.penalty_box_length, (self.width + self.goal_box_width) / 2),  # 12
            (self.penalty_box_length, (self.width + self.penalty_box_width) / 2),  # 13
            (self.length / 2, 0),  # 14
            (self.length / 2, self.width / 2 - self.centre_circle_radius),  # 15
            (self.length / 2, self.width / 2 + self.centre_circle_radius),  # 16
            (self.length / 2, self.width),  # 17
            (self.length - self.penalty_box_length, (self.width - self.penalty_box_width) / 2),  # 18
            (self.length - self.penalty_box_length, (self.width - self.goal_box_width) / 2),  # 19
            (self.length - self.penalty_box_length, (self.width + self.goal_box_width) / 2),  # 20
            (self.length - self.penalty_box_length, (self.width + self.penalty_box_width) / 2),  # 21
            (self.length - self.penalty_spot_distance, self.width / 2),  # 22
            (self.length - self.goal_box_length, (self.width - self.goal_box_width) / 2),  # 23
            (self.length - self.goal_box_length, (self.width + self.goal_box_width) / 2),  # 24
            (self.length, 0),  # 25
            (self.length, (self.width - self.penalty_box_width) / 2),  # 26
            (self.length, (self.width - self.goal_box_width) / 2),  # 27
            (self.length, (self.width + self.goal_box_width) / 2),  # 28
            (self.length, (self.width + self.penalty_box_width) / 2),  # 29
            (self.length, self.width),  # 30
            (self.length / 2 - self.centre_circle_radius, self.width / 2),  # 31
            (self.length / 2 + self.centre_circle_radius, self.width / 2),  # 32
        ]

class ViewTransformer():
    def __init__(self, model_pitch_path="models/best1_pitch.pt"):
        # Load the pitch keypoint pose estimation model
        self.model = YOLO(model_pitch_path)
        
        # Configure standard 2D pitch vertices in meters
        self.pitch_config = SoccerPitchConfiguration()
        self.pitch_vertices = np.array(self.pitch_config.vertices, dtype=np.float32)
        
        # Keep original static vertices as a fallback
        self.pixel_vertices = np.array([[110, 1035], 
                               [265, 275], 
                               [910, 260], 
                               [1640, 915]], dtype=np.float32)
        
        self.target_vertices = np.array([
            [0, 68.0],
            [0, 0.0],
            [23.32, 0.0],
            [23.32, 68.0]
        ], dtype=np.float32)
        
        self.default_H = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point, H):
        if H is None:
            return None
        reshaped_point = np.array(point, dtype=np.float32).reshape(-1, 1, 2)
        transformed_point = cv2.perspectiveTransform(reshaped_point, H)
        return transformed_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks, video_frames=None, read_from_stub=True, stub_path="stubs/homography_stubs.pkl"):
        # 1. Load or estimate homography matrices per frame
        H_per_frame = []
        
        # Load from cache stub if available
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                H_per_frame = pickle.load(f)
            print(f"Loaded dynamic homography matrices from cache: {stub_path}")
        else:
            if video_frames is None:
                raise ValueError("video_frames is required to estimate dynamic homography matrices.")
            
            print("Estimating dynamic homography matrices per frame using YOLO Pose model...")
            last_valid_H = self.default_H
            
            # Predict pose keypoints on each frame
            for frame_num, frame in enumerate(video_frames):
                results = self.model(frame)
                result = results[0]
                H = None
                
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    kp = result.keypoints
                    if len(kp.xy) > 0:
                        # Extract first detected pitch keypoints
                        xy = kp.xy[0].cpu().numpy()
                        conf = kp.conf[0].cpu().numpy()
                        
                        # Filter keypoints where confidence is greater than 0.9
                        valid_mask = conf > 0.9
                        src_pts = xy[valid_mask]
                        dst_pts = self.pitch_vertices[valid_mask]
                        
                        if len(src_pts) >= 4:
                            # Solve perspective transform matrix H with robust RANSAC
                            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                            if H is not None:
                                last_valid_H = H
                
                # If frame has insufficient keypoints, fall back to last valid frame H
                if H is None:
                    H = last_valid_H
                
                H_per_frame.append(H)
            
            # Save estimated matrices to cache stub
            if stub_path is not None:
                os.makedirs(os.path.dirname(stub_path), exist_ok=True)
                with open(stub_path, 'wb') as f:
                    pickle.dump(H_per_frame, f)
                print(f"Saved dynamic homography matrices to cache: {stub_path}")

        # 2. Transform the position coordinates in tracks using per-frame homography
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                H = H_per_frame[frame_num] if frame_num < len(H_per_frame) else self.default_H
                for track_id, track_info in track.items():
                    # Retrieve adjusted position (or fallback to normal position)
                    position = track_info.get('position_adjusted', None)
                    if position is None:
                        position = track_info.get('position', None)
                    
                    if position is None:
                        tracks[object_type][frame_num][track_id]['position_transformed'] = None
                        continue
                    
                    # Convert to numpy array and perform perspective transform
                    position = np.array(position, dtype=np.float32)
                    position_transformed = self.transform_point(position, H)
                    
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    
                    tracks[object_type][frame_num][track_id]['position_transformed'] = position_transformed