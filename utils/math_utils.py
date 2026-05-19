import numpy as np
import cv2

def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two 2D points."""
    if p1 is None or p2 is None:
        return 0.0
    return float(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))

def calculate_centroid(points):
    """Calculates the centroid (mean X, mean Y) of a list of points."""
    if not points:
        return None
    pts = np.array(points)
    centroid = np.mean(pts, axis=0)
    return float(centroid[0]), float(centroid[1])

def calculate_convex_hull_area(points):
    """Calculates the area of the convex hull enclosing the points.
    Uses OpenCV's convexHull and contourArea for efficiency and compatibility.
    """
    if len(points) < 3:
        return 0.0
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(pts)
    area = cv2.contourArea(hull)
    return float(area)

def get_relative_zone(pos, attack_direction='left_to_right'):
    """Determines the tactical zone of the player on a 120m x 70m pitch,
    normalized by their attack direction (Defensive Third, Middle Third, Attacking Third).
    """
    if pos is None:
        return "Unknown"
    
    x, y = pos
    
    # 1. Horizontal zone (relative to attack direction)
    if attack_direction == 'left_to_right':
        if x < 40:
            horizontal_zone = "Defensive Third"
        elif x < 80:
            horizontal_zone = "Middle Third"
        else:
            horizontal_zone = "Attacking Third"
    else:  # right_to_left
        if x > 80:
            horizontal_zone = "Defensive Third"
        elif x > 40:
            horizontal_zone = "Middle Third"
        else:
            horizontal_zone = "Attacking Third"
            
    # 2. Vertical channel
    if y < 20:
        vertical_channel = "Left Wing" if attack_direction == 'left_to_right' else "Right Wing"
    elif y < 50:
        vertical_channel = "Center Channel"
    else:
        vertical_channel = "Right Wing" if attack_direction == 'left_to_right' else "Left Wing"
        
    return f"{horizontal_zone} ({vertical_channel})"
