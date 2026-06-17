import cv2
import numpy as np

class PitchRenderer:
    def __init__(self, width=1200, height=700, margin=50):
        self.width = width
        self.height = height
        self.margin = margin
        
        # Calculate pitch play area dimensions
        self.play_width = self.width - 2 * self.margin
        self.play_height = self.height - 2 * self.margin
        
        # Real pitch dimensions: 120m x 70m
        self.real_length = 120.0
        self.real_width = 70.0
        
        # Scale: pixels per meter
        self.scale_x = self.play_width / self.real_length
        self.scale_y = self.play_height / self.real_width
        
    def meter_to_pixel(self, x_meter, y_meter):
        """Converts real-world pitch coordinates (meters) to pixel coordinates on the image."""
        # Clamp coordinates to stay on the pitch visual area
        x_pixel = int(self.margin + x_meter * self.scale_x)
        y_pixel = int(self.margin + y_meter * self.scale_y)
        return x_pixel, y_pixel

    def draw_pitch(self):
        """Renders a sleek dark mode soccer pitch."""
        # Sleek dark navy/slate grey background: #0f172a
        bg_color = (40, 23, 15)  # BGR format: Deep Dark Blue/Navy
        # Elegant bright mint-green/white lines: #a7f3d0
        line_color = (208, 243, 225)  # BGR format
        line_thickness = 2
        
        # Create background canvas
        pitch_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        pitch_img[:] = bg_color
        
        # 1. Pitch Outer boundary
        top_left = (self.margin, self.margin)
        bottom_right = (self.width - self.margin, self.height - self.margin)
        cv2.rectangle(pitch_img, top_left, bottom_right, line_color, line_thickness, lineType=cv2.LINE_AA)
        
        # 2. Half-way Line
        center_x = self.width // 2
        cv2.line(pitch_img, (center_x, self.margin), (center_x, self.height - self.margin), line_color, line_thickness, lineType=cv2.LINE_AA)
        
        # 3. Center Circle & Center Spot
        center_y = self.height // 2
        center_circle_radius_meter = 9.15
        center_circle_radius_px = int(center_circle_radius_meter * self.scale_x)
        cv2.circle(pitch_img, (center_x, center_y), center_circle_radius_px, line_color, line_thickness, lineType=cv2.LINE_AA)
        cv2.circle(pitch_img, (center_x, center_y), 5, line_color, -1, lineType=cv2.LINE_AA)
        
        # 4. Left Side Penalty Area
        # Dimensions: 16.5m (penalty box length) x 40.32m (penalty box width)
        # In meters: length=20.15m, width=41.0m
        pen_length_px = int(20.15 * self.scale_x)
        pen_width_half_px = int((41.0 / 2) * self.scale_y)
        
        left_pen_top_left = (self.margin, center_y - pen_width_half_px)
        left_pen_bottom_right = (self.margin + pen_length_px, center_y + pen_width_half_px)
        cv2.rectangle(pitch_img, left_pen_top_left, left_pen_bottom_right, line_color, line_thickness, lineType=cv2.LINE_AA)
        
        # 5. Right Side Penalty Area
        right_pen_top_left = (self.width - self.margin - pen_length_px, center_y - pen_width_half_px)
        right_pen_bottom_right = (self.width - self.margin, center_y + pen_width_half_px)
        cv2.rectangle(pitch_img, right_pen_top_left, right_pen_bottom_right, line_color, line_thickness, lineType=cv2.LINE_AA)
        
        # 6. Goal Areas (5.5m box)
        # Dimensions: length=5.50m, width=18.32m
        goal_length_px = int(5.50 * self.scale_x)
        goal_width_half_px = int((18.32 / 2) * self.scale_y)
        
        # Left Goal Box
        left_goal_top_left = (self.margin, center_y - goal_width_half_px)
        left_goal_bottom_right = (self.margin + goal_length_px, center_y + goal_width_half_px)
        cv2.rectangle(pitch_img, left_goal_top_left, left_goal_bottom_right, line_color, line_thickness, lineType=cv2.LINE_AA)
        
        # Right Goal Box
        right_goal_top_left = (self.width - self.margin - goal_length_px, center_y - goal_width_half_px)
        right_goal_bottom_right = (self.width - self.margin, center_y + goal_width_half_px)
        cv2.rectangle(pitch_img, right_goal_top_left, right_goal_bottom_right, line_color, line_thickness, lineType=cv2.LINE_AA)
        
        # 7. Penalty Spots
        # Left Spot (11m from goal line)
        left_spot_x = int(self.margin + 11.00 * self.scale_x)
        cv2.circle(pitch_img, (left_spot_x, center_y), 4, line_color, -1, lineType=cv2.LINE_AA)
        
        # Right Spot (11m from goal line)
        right_spot_x = int(self.width - self.margin - 11.00 * self.scale_x)
        cv2.circle(pitch_img, (right_spot_x, center_y), 4, line_color, -1, lineType=cv2.LINE_AA)
        
        # 8. Penalty Arcs (D-Box)
        # Radius 9.15m centered at the penalty spots
        arc_radius_px = int(9.15 * self.scale_x)
        
        # Left Arc (drawn from X angle of penalty box length)
        left_pen_edge_x = self.margin + pen_length_px
        cv2.ellipse(pitch_img, (left_spot_x, center_y), (arc_radius_px, arc_radius_px), 0, -53, 53, line_color, line_thickness, lineType=cv2.LINE_AA)
        
        # Right Arc
        right_pen_edge_x = self.width - self.margin - pen_length_px
        cv2.ellipse(pitch_img, (right_spot_x, center_y), (arc_radius_px, arc_radius_px), 0, 127, 233, line_color, line_thickness, lineType=cv2.LINE_AA)
        
        # 9. Corner Arcs (1m radius at all 4 corners)
        corner_r_px = int(1.0 * self.scale_x)
        # Top-Left
        cv2.ellipse(pitch_img, (self.margin, self.margin), (corner_r_px, corner_r_px), 0, 0, 90, line_color, line_thickness, lineType=cv2.LINE_AA)
        # Bottom-Left
        cv2.ellipse(pitch_img, (self.margin, self.height - self.margin), (corner_r_px, corner_r_px), 0, 270, 360, line_color, line_thickness, lineType=cv2.LINE_AA)
        # Top-Right
        cv2.ellipse(pitch_img, (self.width - self.margin, self.margin), (corner_r_px, corner_r_px), 0, 90, 180, line_color, line_thickness, lineType=cv2.LINE_AA)
        # Bottom-Right
        cv2.ellipse(pitch_img, (self.width - self.margin, self.height - self.margin), (corner_r_px, corner_r_px), 0, 180, 270, line_color, line_thickness, lineType=cv2.LINE_AA)
        
        return pitch_img

if __name__ == "__main__":
    renderer = PitchRenderer()
    pitch = renderer.draw_pitch()
    cv2.imwrite("pitch_test.png", pitch)
    print("Pitch image rendered successfully.")
