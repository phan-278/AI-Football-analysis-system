import cv2
import numpy as np
import os

class HeatmapGenerator:
    def __init__(self, width=1200, height=700):
        self.width = width
        self.height = height
        self.scale_x = width / 120.0
        self.scale_y = height / 70.0

    def create_pitch_layout(self):
        """Draws a minimalist, premium dark slate tactical pitch layout."""
        # Slate background
        pitch = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        pitch[:] = (24, 18, 15)  # dark slate blue/gray
        
        # Color for pitch lines (semi-transparent gray/white)
        line_color = (160, 160, 160)
        thick = 2
        
        # Outer boundary
        cv2.rectangle(pitch, (0, 0), (self.width - 1, self.height - 1), line_color, thick)
        
        # Midfield line
        mid_x = int(self.width / 2)
        cv2.line(pitch, (mid_x, 0), (mid_x, self.height), line_color, thick)
        
        # Center circle
        center_circle_r = int(9.15 * self.scale_x)
        cv2.circle(pitch, (mid_x, int(self.height / 2)), center_circle_r, line_color, thick)
        cv2.circle(pitch, (mid_x, int(self.height / 2)), 3, line_color, -1)
        
        # Left penalty area
        pen_width_px = int(41.0 * self.scale_y)
        pen_len_px = int(20.15 * self.scale_x)
        y1_pen = int((self.height - pen_width_px) / 2)
        y2_pen = int((self.height + pen_width_px) / 2)
        cv2.rectangle(pitch, (0, y1_pen), (pen_len_px, y2_pen), line_color, thick)
        
        # Left goal area
        goal_width_px = int(18.32 * self.scale_y)
        goal_len_px = int(5.50 * self.scale_x)
        y1_goal = int((self.height - goal_width_px) / 2)
        y2_goal = int((self.height + goal_width_px) / 2)
        cv2.rectangle(pitch, (0, y1_goal), (goal_len_px, y2_goal), line_color, thick)
        
        # Left penalty spot
        pen_spot_x = int(11.00 * self.scale_x)
        cv2.circle(pitch, (pen_spot_x, int(self.height / 2)), 3, line_color, -1)
        
        # Right penalty area
        y1_pen_r = int((self.height - pen_width_px) / 2)
        y2_pen_r = int((self.height + pen_width_px) / 2)
        cv2.rectangle(pitch, (self.width - pen_len_px, y1_pen_r), (self.width, y2_pen_r), line_color, thick)
        
        # Right goal area
        y1_goal_r = int((self.height - goal_width_px) / 2)
        y2_goal_r = int((self.height + goal_width_px) / 2)
        cv2.rectangle(pitch, (self.width - goal_len_px, y1_goal_r), (self.width, y2_goal_r), line_color, thick)
        
        # Right penalty spot
        cv2.circle(pitch, (self.width - pen_spot_x, int(self.height / 2)), 3, line_color, -1)
        
        return pitch

    def generate_heatmap(self, positions, output_path):
        """Generates a Gaussian heatmap from list of (X, Y) positions in meters and saves as PNG.
        
        Args:
            positions (list of tuples): List of transformed positions [(x1,y1), (x2,y2), ...] in meters.
            output_path (str): File path to save the generated heatmap.
        """
        # Ensure directories exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create base pitch background
        pitch = self.create_pitch_layout()
        
        # Create accumulator grid
        accum = np.zeros((self.height, self.width), dtype=np.float32)
        
        valid_count = 0
        for pos in positions:
            if pos is None:
                continue
            x, y = pos
            px = int(x * self.scale_x)
            py = int(y * self.scale_y)
            
            # Clamp coordinates to grid boundary
            px = max(0, min(self.width - 1, px))
            py = max(0, min(self.height - 1, py))
            
            accum[py, px] += 1.0
            valid_count += 1
            
        if valid_count == 0:
            # If no data, save clean pitch layout
            cv2.imwrite(output_path, pitch)
            return
            
        # Apply Gaussian Blur to create smooth heat zones
        # Kernel size and standard dev can be adjusted based on smoothness requirement
        kernel_size = 51
        blurred = cv2.GaussianBlur(accum, (kernel_size, kernel_size), 0)
        
        # Normalize to 0-255 range
        max_val = np.max(blurred)
        if max_val > 0:
            normalized = (blurred / max_val * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(blurred, dtype=np.uint8)
            
        # Apply color map (JET is standard for hot-to-cold maps)
        heatmap_color = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        
        # Blend the heatmap with the dark pitch line layout
        # Transparent background for 0-intensity pixels, reaching up to 65% opacity
        alpha = (normalized / 255.0) * 0.65
        alpha_3d = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
        
        blended = (pitch * (1 - alpha_3d) + heatmap_color * alpha_3d).astype(np.uint8)
        
        # Save output image
        cv2.imwrite(output_path, blended)
        print(f"Heatmap successfully saved to: {output_path}")
