"""
stats_aggregator/aggregator.py
───────────────────────────────
Đọc tracks dict (output của main.py sau các bước 1-8)
→ Tính toán đầy đủ số liệu thống kê 2 đội
→ Xuất match_stats.json

Data thực tế từ tracker.py / speed_estimator.py:
  tracks["players"][frame_num][track_id] = {
      "bbox":                 [x1,y1,x2,y2],
      "position":             (x_px, y_px),
      "position_adjusted":    (x_px, y_px),
      "position_transformed": (x_m, y_m) hoặc None,
      "speed":                float  (km/h),
      "distance":             float  (meters, tích lũy),
      "team":                 1 hoặc 2,
      "team_color":           (B, G, R),
      "has_ball":             True/False
  }
  team_ball_control: np.array([1,1,2,1,...])  # mỗi frame → team nào giữ bóng

Ghi chú:
  - ViewTransformer dùng sân 68 × 23.32m (một góc camera, không phải toàn sân)
  - distance trong tracks tính bằng MÉT → convert sang km khi xuất
  - frame_rate = 24 fps (cố định trong SpeedAndDistance_Estimator)
"""

import json
import os
import numpy as np
from collections import defaultdict, Counter
from typing import Any


FRAME_RATE = 24   # fps — cố định trong speed_estimator

# Sân thực tế từ ViewTransformer: 68 × 23.32m
# Chia zone theo trục X (chiều dài 23.32m)
FIELD_LENGTH = 23.32   # m (chiều dài vùng camera)
FIELD_WIDTH  = 68.0    # m

# Zone boundary theo chiều X (chia 3 phần bằng nhau)
ZONE_DEF_END = FIELD_LENGTH * 0.33
ZONE_MID_END = FIELD_LENGTH * 0.67


class StatsAggregator:
    """
    Tổng hợp số liệu thống kê từ tracks dict.

    Usage:
        agg = StatsAggregator(
            tracks=tracks,
            team_ball_control=team_ball_control,
            video_path="input_videos/TestVideo1.mp4",
            fps=24
        )
        stats = agg.compute()
        agg.save("outputs/match_stats.json")
    """

    def __init__(
        self,
        tracks: dict[str, Any],
        team_ball_control: np.ndarray,
        video_path: str = "",
        fps: float = 24.0,
    ):
        self.tracks           = tracks
        self.team_ball_control = np.array(team_ball_control)
        self.video_path       = video_path
        self.fps              = fps
        self._stats: dict | None = None

    # ──────────────────────────────────────────────
    # Public
    # ──────────────────────────────────────────────

    def compute(self) -> dict[str, Any]:
        """Tính toán toàn bộ và trả về stats dict."""
        player_tracks = self.tracks.get("players", [])
        total_frames  = len(player_tracks)
        duration_sec  = total_frames / self.fps

        # Tách player data theo team
        team_data = self._split_by_team(player_tracks)

        self._stats = {
            "match_info": {
                "video_file":       os.path.basename(self.video_path),
                "duration_seconds": round(duration_sec, 2),
                "fps":              self.fps,
                "total_frames":     total_frames,
            },
            "ball": self._compute_possession(),
            "team1": self._compute_team_stats(team_data[1], player_tracks, team_id=1),
            "team2": self._compute_team_stats(team_data[2], player_tracks, team_id=2),
        }
        return self._stats

    def save(self, output_path: str) -> str:
        """Lưu stats ra JSON. Tự động compute nếu chưa có."""
        if self._stats is None:
            self.compute()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self._stats, f, ensure_ascii=False, indent=2)
        return os.path.abspath(output_path)

    # ──────────────────────────────────────────────
    # Tách player theo team
    # ──────────────────────────────────────────────

    def _split_by_team(self, player_tracks: list) -> dict[int, dict]:
        """
        Trả về {1: {track_id: [track_info_per_frame]},
                2: {track_id: [track_info_per_frame]}}
        """
        team_data: dict[int, dict] = {1: defaultdict(list), 2: defaultdict(list)}
        for frame_data in player_tracks:
            for tid, info in frame_data.items():
                team = info.get("team")
                if team in (1, 2):
                    team_data[team][tid].append(info)
        return team_data

    # ──────────────────────────────────────────────
    # Possession
    # ──────────────────────────────────────────────

    def _compute_possession(self) -> dict:
        total = len(self.team_ball_control)
        if total == 0:
            return {"possession_team1_pct": 50.0, "possession_team2_pct": 50.0}
        t1 = int(np.sum(self.team_ball_control == 1))
        t2 = int(np.sum(self.team_ball_control == 2))
        return {
            "possession_team1_pct": round(t1 / total * 100, 1),
            "possession_team2_pct": round(t2 / total * 100, 1),
        }

    # ──────────────────────────────────────────────
    # Team stats
    # ──────────────────────────────────────────────

    def _compute_team_stats(
        self,
        team_players: dict[int, list],
        all_player_tracks: list,
        team_id: int,
    ) -> dict:
        if not team_players:
            return {}

        # ── Per-player stats ────────────────────────────────────────────
        players_stats: dict[str, Any] = {}
        all_speeds: list[float] = []
        all_distances_km: list[float] = []

        for tid, frames in team_players.items():
            pstat = self._compute_player_stats(frames)
            players_stats[str(tid)] = pstat
            if pstat["speed_samples"]:
                all_speeds.extend(pstat["speed_samples"])
            all_distances_km.append(pstat["distance_km"])

        # ── Team-level aggregation ───────────────────────────────────────
        total_distance_km = round(sum(all_distances_km), 3)
        avg_speed   = round(float(np.mean(all_speeds)), 2)    if all_speeds else 0.0
        max_speed   = round(float(np.max(all_speeds)), 2)     if all_speeds else 0.0

        # Possession %
        ball = self._compute_possession()
        poss_pct = ball[f"possession_team{team_id}_pct"]

        # Formation history + dominant
        formation_history, dominant_formation = self._compute_formation(
            all_player_tracks, team_id
        )

        # Zone distribution (% thời gian cầu thủ ở mỗi zone)
        zone_dist = self._compute_zone_distribution(team_players)

        # Compactness (khoảng cách trung bình giữa các cầu thủ cùng team)
        avg_compactness = self._compute_compactness(all_player_tracks, team_id)

        # Pressing events (số frame có cầu thủ đang pressing)
        pressing_events = self._compute_pressing_events(all_player_tracks, team_id)

        # Clean players_stats (bỏ speed_samples để JSON gọn)
        for p in players_stats.values():
            p.pop("speed_samples", None)

        return {
            "possession_pct":       poss_pct,
            "total_distance_km":    total_distance_km,
            "avg_speed_kmh":        avg_speed,
            "max_speed_kmh":        max_speed,
            "dominant_formation":   dominant_formation,
            "formation_history":    formation_history,
            "avg_compactness_m":    avg_compactness,
            "pressing_events":      pressing_events,
            "zone_distribution":    zone_dist,
            "players":              players_stats,
        }

    # ──────────────────────────────────────────────
    # Player stats
    # ──────────────────────────────────────────────

    def _compute_player_stats(self, frames: list) -> dict:
        speeds: list[float] = []
        distance_m = 0.0
        ball_touches = 0
        zone_counts: dict[str, int] = {"defensive": 0, "middle": 0, "attacking": 0}

        for info in frames:
            sp = info.get("speed")
            if sp is not None and sp > 0:
                speeds.append(float(sp))

            # distance: lấy giá trị cuối cùng (tích lũy từ speed_estimator)
            dist = info.get("distance")
            if dist is not None:
                distance_m = max(distance_m, float(dist))

            if info.get("has_ball"):
                ball_touches += 1

            pos = info.get("position_transformed")
            if pos is not None:
                zone = self._get_zone(pos[0])
                zone_counts[zone] += 1

        total_zone = sum(zone_counts.values()) or 1
        zone_dist = {k: round(v / total_zone * 100, 1) for k, v in zone_counts.items()}

        return {
            "distance_km":       round(distance_m / 1000, 3),
            "avg_speed_kmh":     round(float(np.mean(speeds)), 2) if speeds else 0.0,
            "max_speed_kmh":     round(float(np.max(speeds)), 2)  if speeds else 0.0,
            "ball_touches":      ball_touches,
            "zone_distribution": zone_dist,
            "speed_samples":     speeds,   # tạm thời, sẽ bị pop sau
        }

    # ──────────────────────────────────────────────
    # Formation detection
    # ──────────────────────────────────────────────

    def _compute_formation(
        self,
        all_player_tracks: list,
        team_id: int,
        sample_every: int = 24,   # mỗi 1 giây lấy 1 snapshot
    ) -> tuple[list[str], str]:
        """
        Phát hiện sơ đồ chiến thuật bằng cách phân tích
        vị trí cầu thủ theo zone (phòng thủ / giữa / tấn công).
        Trả về (formation_history, dominant_formation).
        """
        formation_history: list[str] = []

        for frame_num in range(0, len(all_player_tracks), sample_every):
            frame_data = all_player_tracks[frame_num]
            positions: list[float] = []

            for tid, info in frame_data.items():
                if info.get("team") != team_id:
                    continue
                pos = info.get("position_transformed")
                if pos is not None:
                    positions.append(float(pos[0]))   # trục X = độ sâu trên sân

            if len(positions) < 4:
                continue

            formation = self._positions_to_formation(positions)
            formation_history.append(formation)

        if not formation_history:
            return [], "N/A"

        dominant = Counter(formation_history).most_common(1)[0][0]
        # Giữ tối đa 10 snapshot để JSON không quá dài
        step = max(1, len(formation_history) // 10)
        return formation_history[::step][:10], dominant

    def _positions_to_formation(self, x_positions: list[float]) -> str:
        """
        Phân loại cầu thủ theo zone X → đếm → suy ra sơ đồ.
        Zone dựa trên ViewTransformer: sân 23.32m
          defensive: x < 7.7m
          middle:    7.7 ≤ x < 15.5m
          attacking: x ≥ 15.5m
        """
        def_count = sum(1 for x in x_positions if x < ZONE_DEF_END)
        att_count = sum(1 for x in x_positions if x >= ZONE_MID_END)
        mid_count = len(x_positions) - def_count - att_count

        # Heuristic mapping (dựa trên 10 cầu thủ outfield)
        formations = {
            (4, 3, 3): "4-3-3",
            (4, 4, 2): "4-4-2",
            (4, 2, 4): "4-2-4",
            (3, 5, 2): "3-5-2",
            (3, 4, 3): "3-4-3",
            (5, 3, 2): "5-3-2",
            (5, 4, 1): "5-4-1",
            (4, 5, 1): "4-5-1",
        }
        key = (def_count, mid_count, att_count)
        # Tìm formation gần nhất (theo khoảng cách Manhattan)
        best = min(
            formations.items(),
            key=lambda item: abs(item[0][0] - def_count)
                           + abs(item[0][1] - mid_count)
                           + abs(item[0][2] - att_count),
        )
        return best[1]

    # ──────────────────────────────────────────────
    # Zone distribution
    # ──────────────────────────────────────────────

    def _compute_zone_distribution(
        self, team_players: dict[int, list]
    ) -> dict[str, float]:
        counts = {"defensive": 0, "middle": 0, "attacking": 0}
        for frames in team_players.values():
            for info in frames:
                pos = info.get("position_transformed")
                if pos is not None:
                    counts[self._get_zone(pos[0])] += 1
        total = sum(counts.values()) or 1
        return {k: round(v / total * 100, 1) for k, v in counts.items()}

    def _get_zone(self, x_m: float) -> str:
        if x_m < ZONE_DEF_END:
            return "defensive"
        elif x_m < ZONE_MID_END:
            return "middle"
        else:
            return "attacking"

    # ──────────────────────────────────────────────
    # Compactness
    # ──────────────────────────────────────────────

    def _compute_compactness(
        self,
        all_player_tracks: list,
        team_id: int,
        sample_every: int = 12,
    ) -> float:
        """
        Compactness = khoảng cách trung bình giữa tất cả cặp cầu thủ
        cùng team (tính bằng mét, dùng position_transformed).
        """
        compactness_values: list[float] = []

        for frame_num in range(0, len(all_player_tracks), sample_every):
            frame_data = all_player_tracks[frame_num]
            positions: list[tuple] = []

            for _, info in frame_data.items():
                if info.get("team") != team_id:
                    continue
                pos = info.get("position_transformed")
                if pos is not None:
                    positions.append(tuple(pos))

            if len(positions) < 2:
                continue

            distances: list[float] = []
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    d = float(np.sqrt(
                        (positions[i][0] - positions[j][0]) ** 2 +
                        (positions[i][1] - positions[j][1]) ** 2
                    ))
                    distances.append(d)

            if distances:
                compactness_values.append(float(np.mean(distances)))

        if not compactness_values:
            return 0.0
        return round(float(np.mean(compactness_values)), 2)

    # ──────────────────────────────────────────────
    # Pressing events
    # ──────────────────────────────────────────────

    def _compute_pressing_events(
        self,
        all_player_tracks: list,
        team_id: int,
        press_distance_m: float = 5.0,   # khoảng cách tính là pressing (m)
        min_duration_frames: int = 3,     # pressing phải kéo dài ít nhất N frame
    ) -> int:
        """
        Đếm số lần pressing:
        Cầu thủ team A (không có bóng) tiến vào trong vòng press_distance_m
        của cầu thủ team B (có bóng), kéo dài ít nhất min_duration_frames.
        """
        pressing_streak = 0
        event_count     = 0
        opponent_team   = 3 - team_id  # 1→2, 2→1

        for frame_data in all_player_tracks:
            # Lấy vị trí người có bóng của đội đối phương
            opp_ball_pos = None
            for _, info in frame_data.items():
                if info.get("team") == opponent_team and info.get("has_ball"):
                    opp_ball_pos = info.get("position_transformed")
                    break

            if opp_ball_pos is None:
                pressing_streak = 0
                continue

            # Kiểm tra cầu thủ team_id có ai gần opp_ball_pos không
            is_pressing = False
            for _, info in frame_data.items():
                if info.get("team") != team_id:
                    continue
                pos = info.get("position_transformed")
                if pos is None:
                    continue
                dist = float(np.sqrt(
                    (pos[0] - opp_ball_pos[0]) ** 2 +
                    (pos[1] - opp_ball_pos[1]) ** 2
                ))
                if dist <= press_distance_m:
                    is_pressing = True
                    break

            if is_pressing:
                pressing_streak += 1
                if pressing_streak == min_duration_frames:
                    event_count += 1
            else:
                pressing_streak = 0

        return event_count
