import os
import json
import numpy as np
from .player_stats import PlayerStatsAggregator
from .team_stats import TeamStatsAggregator
from .formation_detector import FormationDetector
from .heatmap_generator import HeatmapGenerator

class StatsExporter:
    def __init__(self, frame_rate=24, output_dir="outputs"):
        self.frame_rate = frame_rate
        self.output_dir = output_dir
        self.heatmaps_dir = os.path.join(output_dir, "heatmaps")
        
        self.player_aggregator = PlayerStatsAggregator(frame_rate)
        self.team_aggregator = TeamStatsAggregator()
        self.formation_detector = FormationDetector(frame_rate)
        self.heatmap_generator = HeatmapGenerator()

    def determine_team_directions(self, tracks):
        """Determines the default attacking direction of each team based on average player locations."""
        x_coords_team1 = []
        x_coords_team2 = []
        
        for player_track in tracks.get('players', []):
            for info in player_track.values():
                team = info.get('team')
                pos = info.get('position_transformed')
                if team == 1 and pos is not None:
                    x_coords_team1.append(pos[0])
                elif team == 2 and pos is not None:
                    x_coords_team2.append(pos[0])
                    
        mean_x1 = np.mean(x_coords_team1) if x_coords_team1 else 40.0
        mean_x2 = np.mean(x_coords_team2) if x_coords_team2 else 80.0
        
        # The team with the lower average X-coordinate defends the Left half (x=0) and attacks Right (x=120)
        if mean_x1 < mean_x2:
            return {
                1: 'left_to_right',
                2: 'right_to_left'
            }
        else:
            return {
                1: 'right_to_left',
                2: 'left_to_right'
            }

    def process_and_export(self, tracks, team_ball_control, output_json_filename="match_stats.json"):
        """Orchestrates the entire statistical analysis, generates heatmaps, and saves data to a JSON file.
        
        Args:
            tracks (dict): The complete tracks dictionary.
            team_ball_control (list or np.ndarray): Control history of the ball.
            output_json_filename (str): Name of the output JSON file.
            
        Returns:
            dict: The complete aggregated statistics.
        """
        # Ensure directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.heatmaps_dir, exist_ok=True)
        
        total_frames = len(tracks.get('players', []))
        match_duration = float(total_frames / self.frame_rate)
        
        # 1. Determine Team Directions
        team_directions = self.determine_team_directions(tracks)
        
        # 2. Player Stats
        print("Aggregating player statistics...")
        player_stats = self.player_aggregator.aggregate_player_stats(tracks, team_directions)
        
        # 3. Team Stats
        print("Aggregating team statistics...")
        team_stats = self.team_aggregator.calculate_team_stats(tracks, team_ball_control, player_stats)
        
        # 4. Formation Detection
        print("Detecting team formations...")
        formations = self.formation_detector.detect_formations(tracks, team_directions)
        
        # Add formations to team stats
        for team in [1, 2]:
            team_stats[team]["overall_formation"] = formations["overall"][team]
            team_stats[team]["formation_history"] = formations["windowed"][team]

        # 5. Generate Heatmaps
        print("Generating heatmaps...")
        # Gather position histories
        player_positions = {}
        team_positions = {1: [], 2: []}
        
        for player_track in tracks.get('players', []):
            for player_id, info in player_track.items():
                pos = info.get('position_transformed')
                team = info.get('team')
                if pos is not None:
                    if player_id not in player_positions:
                        player_positions[player_id] = []
                    player_positions[player_id].append(pos)
                    if team in [1, 2]:
                        team_positions[team].append(pos)

        # Generate player heatmaps
        for player_id, positions in player_positions.items():
            if player_id in player_stats:
                filename = f"player_{player_id}.png"
                output_path = os.path.join(self.heatmaps_dir, filename)
                self.heatmap_generator.generate_heatmap(positions, output_path)
                # Save relative path for easy frontend/NLP loading
                player_stats[player_id]["heatmap_path"] = os.path.join("outputs", "heatmaps", filename).replace("\\", "/")

        # Generate team heatmaps
        for team in [1, 2]:
            filename = f"team_{team}.png"
            output_path = os.path.join(self.heatmaps_dir, filename)
            self.heatmap_generator.generate_heatmap(team_positions[team], output_path)
            team_stats[team]["heatmap_path"] = os.path.join("outputs", "heatmaps", filename).replace("\\", "/")

        # 6. Gather all into a clean match report dictionary
        match_stats = {
            "match_summary": {
                "total_frames": total_frames,
                "frame_rate": self.frame_rate,
                "match_duration_seconds": match_duration
            },
            "team_directions": {
                "team_1": team_directions[1],
                "team_2": team_directions[2]
            },
            "teams": {
                "team_1": team_stats[1],
                "team_2": team_stats[2]
            },
            "players": list(player_stats.values())
        }

        # Save to JSON file
        json_output_path = os.path.join(self.output_dir, output_json_filename)
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(match_stats, f, indent=4, ensure_ascii=False)
            
        print(f"Match statistics exported successfully to: {json_output_path}")
        return match_stats
