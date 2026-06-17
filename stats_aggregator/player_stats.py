import numpy as np
from utils import get_relative_zone

class PlayerStatsAggregator:
    def __init__(self, frame_rate=24):
        self.frame_rate = frame_rate

    def aggregate_player_stats(self, tracks, team_directions):
        """Aggregates statistics for each player from the tracking data.
        
        Args:
            tracks (dict): The complete tracks dictionary.
            team_directions (dict): Dict mapping team ID (1 or 2) to their attack direction ('left_to_right' or 'right_to_left').
            
        Returns:
            dict: Nested dictionary containing stats for each player, keyed by player_id.
        """
        player_frames_data = {}
        player_teams = {}
        
        # 1. Gather all frame-by-frame raw data for each player
        for frame_num, player_track in enumerate(tracks.get('players', [])):
            for player_id, info in player_track.items():
                if player_id not in player_frames_data:
                    player_frames_data[player_id] = {
                        'speeds': [],
                        'distances': [],
                        'positions': [],
                        'possession_count': 0,
                        'total_frames': 0
                    }
                
                # Record team of the player (should be constant, but we take it dynamically)
                team = info.get('team')
                if team is not None:
                    player_teams[player_id] = team
                
                speed = info.get('speed')
                if speed is not None:
                    player_frames_data[player_id]['speeds'].append(speed)
                    
                distance = info.get('distance')
                if distance is not None:
                    player_frames_data[player_id]['distances'].append(distance)
                    
                pos = info.get('position_transformed')
                if pos is not None:
                    player_frames_data[player_id]['positions'].append(pos)
                    
                if info.get('has_ball', False):
                    player_frames_data[player_id]['possession_count'] += 1
                    
                player_frames_data[player_id]['total_frames'] += 1

        # 2. Compute aggregate metrics for each player
        aggregated_stats = {}
        for player_id, data in player_frames_data.items():
            team = player_teams.get(player_id, 1)
            direction = team_directions.get(team, 'left_to_right')
            
            # Distance
            total_dist = max(data['distances']) if data['distances'] else 0.0
            
            # Speed
            avg_speed = np.mean(data['speeds']) if data['speeds'] else 0.0
            max_speed = max(data['speeds']) if data['speeds'] else 0.0
            
            # Possession
            possession_sec = data['possession_count'] / self.frame_rate
            
            # Zones
            zones = [get_relative_zone(pos, direction) for pos in data['positions'] if pos is not None]
            if zones:
                # Find most frequent zone
                unique_zones, counts = np.unique(zones, return_counts=True)
                favorite_zone = unique_zones[np.argmax(counts)]
                
                # Zone distribution percentage
                total_pos = len(zones)
                zone_distribution = {z: float(c / total_pos * 100) for z, c in zip(unique_zones, counts)}
            else:
                favorite_zone = "Unknown"
                zone_distribution = {}
                
            aggregated_stats[player_id] = {
                "player_id": int(player_id),
                "team": int(team),
                "total_distance_meters": float(total_dist),
                "average_speed_kmh": float(avg_speed),
                "max_speed_kmh": float(max_speed),
                "possession_seconds": float(possession_sec),
                "possession_frames": int(data['possession_count']),
                "favorite_zone": favorite_zone,
                "zone_distribution": zone_distribution
            }
            
        return aggregated_stats
