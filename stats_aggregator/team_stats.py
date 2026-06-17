import numpy as np
from utils import calculate_distance, calculate_centroid, calculate_convex_hull_area

class TeamStatsAggregator:
    def __init__(self):
        pass

    def calculate_team_stats(self, tracks, team_ball_control, player_stats):
        """Calculates stats for Team 1 and Team 2.
        
        Args:
            tracks (dict): Complete tracks dictionary.
            team_ball_control (np.ndarray or list): Array containing the controlling team ID per frame.
            player_stats (dict): Aggregated player stats (used for team total distance and speed).
            
        Returns:
            dict: Stats for Team 1 and Team 2.
        """
        # 1. Team Possession Percentage
        team_ball_control = np.array(team_ball_control)
        total_possession_frames = len(team_ball_control)
        
        possession_pct = {1: 50.0, 2: 50.0}  # default fallback
        if total_possession_frames > 0:
            count_team1 = np.sum(team_ball_control == 1)
            count_team2 = np.sum(team_ball_control == 2)
            total_counted = count_team1 + count_team2
            if total_counted > 0:
                possession_pct[1] = float(count_team1 / total_counted * 100)
                possession_pct[2] = float(count_team2 / total_counted * 100)

        # 2. Team Distance & Speed (derived from player stats)
        team_total_distance = {1: 0.0, 2: 0.0}
        team_speeds = {1: [], 2: []}
        
        for p_id, p_info in player_stats.items():
            team = p_info['team']
            if team in [1, 2]:
                team_total_distance[team] += p_info['total_distance_meters']
                team_speeds[team].append(p_info['average_speed_kmh'])
                
        team_avg_speed = {
            1: float(np.mean(team_speeds[1])) if team_speeds[1] else 0.0,
            2: float(np.mean(team_speeds[2])) if team_speeds[2] else 0.0
        }

        # 3. Compactness & Pressing over time
        team_compactness_hull = {1: [], 2: []}
        team_compactness_spread = {1: [], 2: []}
        
        pressing_distances = {1: [], 2: []}  # distance of defenders to opponent ball carrier
        pressing_frames_within_5m = {1: 0, 2: 0}
        opponent_possession_frames = {1: 0, 2: 0} # frames where the other team has the ball
        
        num_frames = len(tracks.get('players', []))
        
        for frame_num in range(num_frames):
            players_in_frame = tracks['players'][frame_num]
            ball_in_frame = tracks['ball'][frame_num] if frame_num < len(tracks.get('ball', [])) else {}
            
            # Group player positions by team
            team_positions = {1: [], 2: []}
            ball_carrier_pos = None
            ball_carrier_team = None
            
            for player_id, info in players_in_frame.items():
                pos = info.get('position_transformed')
                team = info.get('team')
                if pos is not None and team in [1, 2]:
                    team_positions[team].append(pos)
                    if info.get('has_ball', False):
                        ball_carrier_pos = pos
                        ball_carrier_team = team
                        
            # If no ball carrier was explicitly marked, but ball position is available, use it
            if ball_carrier_pos is None and 1 in ball_in_frame:
                ball_carrier_pos = ball_in_frame[1].get('position_transformed')
                # Determine who is closer to the ball to infer team in possession
                if ball_carrier_pos is not None:
                    # Let's use the team ball control value for this frame
                    ball_carrier_team = team_ball_control[frame_num] if frame_num < len(team_ball_control) else None

            # Calculate compactness for each team in this frame
            for team in [1, 2]:
                positions = team_positions[team]
                if len(positions) >= 3:
                    # Convex Hull Area
                    hull_area = calculate_convex_hull_area(positions)
                    team_compactness_hull[team].append(hull_area)
                    
                    # Spread (avg distance to centroid)
                    centroid = calculate_centroid(positions)
                    if centroid:
                        spreads = [calculate_distance(pos, centroid) for pos in positions]
                        team_compactness_spread[team].append(np.mean(spreads))
                elif len(positions) > 0:
                    centroid = calculate_centroid(positions)
                    if centroid:
                        spreads = [calculate_distance(pos, centroid) for pos in positions]
                        team_compactness_spread[team].append(np.mean(spreads))

            # Calculate pressing for this frame
            # Pressing is defenders approaching the opponent with the ball
            if ball_carrier_pos is not None and ball_carrier_team in [1, 2]:
                defending_team = 2 if ball_carrier_team == 1 else 1
                defenders_pos = team_positions[defending_team]
                
                opponent_possession_frames[defending_team] += 1
                
                if defenders_pos:
                    # Find distance to closest defender
                    dists = [calculate_distance(pos, ball_carrier_pos) for pos in defenders_pos]
                    min_dist = min(dists)
                    pressing_distances[defending_team].append(min_dist)
                    if min_dist <= 5.0:
                        pressing_frames_within_5m[defending_team] += 1

        # 4. Average metrics over the match
        team_stats = {}
        for team in [1, 2]:
            opponent = 2 if team == 1 else 1
            
            # Avg Compactness
            avg_hull = float(np.mean(team_compactness_hull[team])) if team_compactness_hull[team] else 0.0
            avg_spread = float(np.mean(team_compactness_spread[team])) if team_compactness_spread[team] else 0.0
            
            # Avg Pressing
            avg_press_dist = float(np.mean(pressing_distances[team])) if pressing_distances[team] else 0.0
            total_opp_pos = opponent_possession_frames[team]
            pressing_intensity = float(pressing_frames_within_5m[team] / total_opp_pos * 100) if total_opp_pos > 0 else 0.0
            
            team_stats[team] = {
                "team_id": int(team),
                "possession_percentage": possession_pct[team],
                "total_distance_meters": team_total_distance[team],
                "average_speed_kmh": team_avg_speed[team],
                "compactness": {
                    "convex_hull_area_m2": avg_hull,
                    "average_spread_meters": avg_spread
                },
                "pressing": {
                    "average_distance_to_ball_carrier_meters": avg_press_dist,
                    "pressing_intensity_percentage": pressing_intensity
                }
            }
            
        return team_stats
