import numpy as np

class FormationDetector:
    def __init__(self, frame_rate=24, window_seconds=30):
        self.frame_rate = frame_rate
        self.window_seconds = window_seconds
        self.window_frames = int(window_seconds * frame_rate)

    def _kmeans_1d(self, data, k=3, max_iter=20):
        """Simple self-contained 1D K-Means clustering algorithm to avoid external dependencies.
        
        Args:
            data (np.ndarray): 1D array of coordinates.
            k (int): Number of clusters.
            
        Returns:
            np.ndarray: Labels for each point.
            np.ndarray: Final sorted centroids.
        """
        if len(data) < k:
            # Fallback if too few players
            labels = np.arange(len(data)) % k
            centroids = np.linspace(data.min(), data.max(), k) if len(data) > 0 else np.zeros(k)
            return labels, centroids

        data_sorted = np.sort(data)
        # Initialize centroids as percentiles (e.g., 15th, 50th, 85th percentile)
        centroids = np.percentile(data, np.linspace(15, 85, k))
        
        for _ in range(max_iter):
            # Distance of each point to each centroid
            distances = np.abs(data[:, np.newaxis] - centroids)
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros(k)
            for i in range(k):
                cluster_pts = data[labels == i]
                if len(cluster_pts) > 0:
                    new_centroids[i] = np.mean(cluster_pts)
                else:
                    new_centroids[i] = centroids[i]
            
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
            
        # Ensure centroids are sorted and labels match the sorted centroids
        sorted_indices = np.argsort(centroids)
        sorted_centroids = centroids[sorted_indices]
        
        # Map labels to sorted order
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_indices)}
        labels = np.array([label_mapping[label] for label in labels])
        
        return labels, sorted_centroids

    def detect_goalkeeper(self, player_x_coords, direction):
        """Identifies the goalkeeper as the player whose average X position is closest to their own goal line."""
        goalkeeper_id = None
        min_distance_to_goal = float('inf')
        
        for player_id, x_vals in player_x_coords.items():
            if not x_vals:
                continue
            avg_x = np.mean(x_vals)
            
            # Distance to own goal
            if direction == 'left_to_right':
                dist = avg_x  # goal is at x=0
            else:
                dist = 120.0 - avg_x  # goal is at x=120
                
            if dist < min_distance_to_goal:
                min_distance_to_goal = dist
                goalkeeper_id = player_id
                
        return goalkeeper_id

    def detect_formations(self, tracks, team_directions):
        """Detects the formation of each team over the whole match and per 30-second window.
        
        Args:
            tracks (dict): Complete tracks dictionary.
            team_directions (dict): Dict mapping team ID to attack direction.
            
        Returns:
            dict: Detected formation details.
        """
        num_frames = len(tracks.get('players', []))
        
        # 1. Gather all X positions for each player over the entire match
        player_x_coords = {1: {}, 2: {}}
        for player_track in tracks.get('players', []):
            for player_id, info in player_track.items():
                team = info.get('team')
                pos = info.get('position_transformed')
                if team in [1, 2] and pos is not None:
                    if player_id not in player_x_coords[team]:
                        player_x_coords[team][player_id] = []
                    player_x_coords[team][player_id].append(pos[0])

        # 2. Identify Goalkeeper for each team
        goalkeepers = {}
        for team in [1, 2]:
            goalkeepers[team] = self.detect_goalkeeper(player_x_coords[team], team_directions[team])

        # 3. Detect match-wide formation
        overall_formations = {}
        for team in [1, 2]:
            overall_formations[team] = self._detect_formation_for_subset(
                player_x_coords[team], 
                goalkeepers[team], 
                team_directions[team]
            )

        # 4. Detect windowed formations (every 30 seconds)
        windowed_formations = {1: [], 2: []}
        num_windows = int(np.ceil(num_frames / self.window_frames))
        
        for w in range(num_windows):
            start_f = w * self.window_frames
            end_f = min((w + 1) * self.window_frames, num_frames)
            
            # Gather coordinates for this window
            window_coords = {1: {}, 2: {}}
            for f in range(start_f, end_f):
                player_track = tracks['players'][f]
                for player_id, info in player_track.items():
                    team = info.get('team')
                    pos = info.get('position_transformed')
                    if team in [1, 2] and pos is not None:
                        if player_id not in window_coords[team]:
                            window_coords[team][player_id] = []
                        window_coords[team][player_id].append(pos[0])
                        
            # Detect for this window
            for team in [1, 2]:
                form = self._detect_formation_for_subset(
                    window_coords[team], 
                    goalkeepers[team], 
                    team_directions[team]
                )
                windowed_formations[team].append({
                    "window_index": w,
                    "start_time_seconds": float(start_f / self.frame_rate),
                    "end_time_seconds": float(end_f / self.frame_rate),
                    "formation": form
                })
                
        return {
            "overall": overall_formations,
            "windowed": windowed_formations
        }

    def _detect_formation_for_subset(self, player_coords, goalie_id, direction):
        """Helper to detect formation from a subset of player coordinates."""
        avg_x_norm = []
        player_ids = []
        
        for player_id, x_vals in player_coords.items():
            if player_id == goalie_id or not x_vals:
                continue
            
            avg_x = np.mean(x_vals)
            # Normalize so 0 is defending end line and 120 is attacking end line
            avg_x_norm_val = avg_x if direction == 'left_to_right' else 120.0 - avg_x
            avg_x_norm.append(avg_x_norm_val)
            player_ids.append(player_id)
            
        if len(avg_x_norm) < 3:
            return "4-3-3"  # default fallback if too few players are tracked
            
        # Cluster outfield players into 3 lines (Defenders, Midfielders, Attackers)
        avg_x_norm = np.array(avg_x_norm)
        labels, centroids = self._kmeans_1d(avg_x_norm, k=3)
        
        # Count players in each cluster
        # Label 0 represents Defenders (closest to own goal), 1: Midfielders, 2: Attackers
        defenders_count = int(np.sum(labels == 0))
        midfielders_count = int(np.sum(labels == 1))
        attackers_count = int(np.sum(labels == 2))
        
        # Standardize representation
        return f"{defenders_count}-{midfielders_count}-{attackers_count}"
