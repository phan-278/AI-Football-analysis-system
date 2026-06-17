import cv2
import numpy as np
import os
from collections import defaultdict

class TaticalMap:
    def __init__(self, map_path="input_videos/map.webp"):
        self.map_path = map_path
        
        # Load map background
        self.map_bg = cv2.imread(self.map_path)
        if self.map_bg is None:
            print(f"Warning: Map image not found at {self.map_path}. Creating fallback background.")
            # Fallback slate background
            self.map_bg = np.zeros((309, 474, 3), dtype=np.uint8)
            self.map_bg[:] = (40, 23, 15)
            
        self.map_h, self.map_w = self.map_bg.shape[:2]
        
        # Real pitch dimensions for mapping: 120m x 70m
        self.real_length = 120.0
        self.real_width = 70.0
        
        # Scale factors
        self.scale_x = self.map_w / self.real_length
        self.scale_y = self.map_h / self.real_width
        
        # Trail history to save the last N positions (for smooth trail animations)
        self.trail_history = defaultdict(list)
        self.max_trail_len = 15
        
    def meter_to_pixel(self, x_meter, y_meter):
        """Converts real-world coordinates (meters) to map pixels."""
        px = int(x_meter * self.scale_x)
        py = int(y_meter * self.scale_y)
        # Clamp to map bounds
        px = max(0, min(self.map_w - 1, px))
        py = max(0, min(self.map_h - 1, py))
        return px, py

    def draw_tatical_map_pip(self, video_frames, tracks):
        """Draws players/ball and overlays the map on the top-right corner of the video frames."""
        output_frames = []
        
        print("Rendering Tactical Map overlays on top-right of video frames...")
        for frame_num, frame in enumerate(video_frames):
            annotated_frame = frame.copy()
            
            # 1. Start with map background copy
            pitch_img = self.map_bg.copy()
            
            # 2. Draw Trails (fading lines)
            self._update_trail_history(tracks, frame_num)
            self._draw_trails(pitch_img)
            
            # 3. Draw Players
            player_dict = tracks.get('players', [])[frame_num] if frame_num < len(tracks.get('players', [])) else {}
            for track_id, player_info in player_dict.items():
                pos = player_info.get('position_transformed', None)
                if pos is None:
                    continue
                
                px, py = self.meter_to_pixel(pos[0], pos[1])
                team_color = player_info.get('team_color', (0, 0, 255))
                
                # Draw player circle (body) using player's team_color
                cv2.circle(pitch_img, (px, py), 8, team_color, -1, lineType=cv2.LINE_AA)
                # Outer ring for contrast
                cv2.circle(pitch_img, (px, py), 8, (255, 255, 255), 1, lineType=cv2.LINE_AA)
                
                # Draw player ID number inside
                text = str(track_id)
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
                text_x = px - text_size[0] // 2
                text_y = py + text_size[1] // 2
                cv2.putText(pitch_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            
            # 4. Draw Ball
            ball_dict = tracks.get('ball', [])[frame_num] if frame_num < len(tracks.get('ball', [])) else {}
            for _, ball_info in ball_dict.items():
                pos = ball_info.get('position_transformed', None)
                if pos is None:
                    continue
                
                px, py = self.meter_to_pixel(pos[0], pos[1])
                
                # Neon orange ball
                cv2.circle(pitch_img, (px, py), 5, (0, 255, 255), -1, lineType=cv2.LINE_AA)
                cv2.circle(pitch_img, (px, py), 5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
                
            # 5. Overlay the map on the top-right corner of the video frame
            frame_h, frame_w = annotated_frame.shape[:2]
            margin_x = 30
            margin_y = 30
            
            # Top-right coordinates
            y1 = margin_y
            y2 = margin_y + self.map_h
            x1 = frame_w - self.map_w - margin_x
            x2 = frame_w - margin_x
            
            # Draw premium white card border around the map PiP card
            border_thick = 2
            cv2.rectangle(annotated_frame, (x1 - border_thick, y1 - border_thick), (x2 + border_thick, y2 + border_thick), (255, 255, 255), border_thick, lineType=cv2.LINE_AA)
            
            # Overlay
            annotated_frame[y1:y2, x1:x2] = pitch_img
            
            output_frames.append(annotated_frame)
            
        return output_frames

    def _update_trail_history(self, tracks, frame_num):
        """Appends current active object positions to the trail history buffer."""
        # 1. Update player trails
        player_dict = tracks.get('players', [])[frame_num] if frame_num < len(tracks.get('players', [])) else {}
        active_ids = set()
        
        for track_id, player_info in player_dict.items():
            pos = player_info.get('position_transformed', None)
            if pos is not None:
                key = ('player', track_id, tuple(player_info.get('team_color', (0, 0, 255))))
                self.trail_history[key].append(pos)
                active_ids.add(key)
                
                if len(self.trail_history[key]) > self.max_trail_len:
                    self.trail_history[key].pop(0)
                    
        # 2. Update ball trail
        ball_dict = tracks.get('ball', [])[frame_num] if frame_num < len(tracks.get('ball', [])) else {}
        for track_id, ball_info in ball_dict.items():
            pos = ball_info.get('position_transformed', None)
            if pos is not None:
                key = ('ball', track_id, (0, 255, 255))
                self.trail_history[key].append(pos)
                active_ids.add(key)
                
                if len(self.trail_history[key]) > self.max_trail_len:
                    self.trail_history[key].pop(0)

        # 3. Decay inactive trails
        inactive_keys = [k for k in self.trail_history.keys() if k not in active_ids]
        for k in inactive_keys:
            if self.trail_history[k]:
                self.trail_history[k].pop(0)
            else:
                del self.trail_history[k]

    def _draw_trails(self, pitch_img):
        """Draws the fading trajectory trails on the tactical map image."""
        for key, points in self.trail_history.items():
            if len(points) < 2:
                continue
                
            object_type, _, base_color = key
            
            for i in range(len(points) - 1):
                pt_prev = points[i]
                pt_curr = points[i+1]
                
                factor = (i + 1) / len(points)
                faded_color = tuple(int(c * factor) for c in base_color)
                
                p1 = self.meter_to_pixel(pt_prev[0], pt_prev[1])
                p2 = self.meter_to_pixel(pt_curr[0], pt_curr[1])
                
                thickness = max(1, int(2 * factor))
                
                cv2.line(pitch_img, p1, p2, faded_color, thickness, lineType=cv2.LINE_AA)
