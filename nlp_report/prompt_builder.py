import json
from typing import Any


class PromptBuilder:
    """
    Convert match_stats → (system_prompt, user_prompt).

    Usage:
        builder = PromptBuilder(match_stats)
        system_prompt, user_prompt = builder.build()
    """

    SYSTEM_PROMPT = """You are a professional football tactical analyst with over 10 years of experience.
Task: read match statistics extracted by AI Computer Vision
and write an in-depth tactical analysis report in English.

Report requirements:
- Analyze tactical formation and playing style of each team
- Compare pressing, ball possession, space usage
- Point out specific strengths and weaknesses based on data
- Professional language, standard football terminology
- Comments must stick strictly to provided actual numbers

Reply ONLY with valid JSON in the following structure (no markdown, no explanations):
{
  "match_overview": "Match overview in 2-3 sentences, mentioning duration and overall playing style",
  "team1_analysis": {
    "title": "Short title describing team 1 style",
    "formation": "Description of formation and tactical deployment",
    "strengths": ["specific strength 1 based on data", "strength 2", "strength 3"],
    "weaknesses": ["weakness 1", "weakness 2"],
    "tactical_summary": "Detailed tactical analysis in 4-5 sentences, mentioning specific numbers"
  },
  "team2_analysis": {
    "title": "Short title describing team 2 style",
    "formation": "Description of formation and tactical deployment",
    "strengths": ["strength 1", "strength 2", "strength 3"],
    "weaknesses": ["weakness 1", "weakness 2"],
    "tactical_summary": "Detailed tactical analysis in 4-5 sentences, mentioning specific numbers"
  },
  "comparison": {
    "possession_battle": "Detailed comment on possession battle",
    "pressing_duel": "Compare pressing intensity and efficiency of 2 teams based on data",
    "space_usage": "Analyze how 2 teams exploit space, based on zone distribution",
    "physical_comparison": "Physical comparison: total distance, avg speed, max speed",
    "key_difference": "Core tactical difference that decided the match"
  },
  "key_players": [
    {
      "team": "Team 1 or Team 2",
      "player_id": "Player ID",
      "role": "Tactical role",
      "highlight": "Short comment on this player based on data"
    }
  ],
  "conclusion": "Overall conclusion in 3-4 sentences about tactics and match progression"
}"""

    def __init__(self, match_stats: dict[str, Any]):
        self.stats = match_stats

    @classmethod
    def from_json_file(cls, path: str) -> "PromptBuilder":
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def build(self) -> tuple[str, str]:
        """Returns (system_prompt, user_prompt)."""
        return self.SYSTEM_PROMPT, self._build_user_prompt()

    # ──────────────────────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────────────────────

    def _build_user_prompt(self) -> str:
        s    = self.stats
        info = s.get("match_info", {})
        t1   = s.get("team1", {})
        t2   = s.get("team2", {})
        ball = s.get("ball", {})

        dur_min = round(info.get("duration_seconds", 0) / 60, 1)

        lines = [
            "=== MATCH STATISTICS DATA (extracted by AI Computer Vision) ===",
            "",
            f"Video       : {info.get('video_file', 'N/A')}",
            f"Duration    : {dur_min} mins  |  FPS: {info.get('fps', 24)}  |  "
            f"Total frames: {info.get('total_frames', 0)}",
            "",
            "┌─────────────────────────────────────────┐",
            "│              POSSESSION                 │",
            "└─────────────────────────────────────────┘",
            f"  Team 1: {ball.get('possession_team1_pct', 0):.1f}%",
            f"  Team 2: {ball.get('possession_team2_pct', 0):.1f}%",
            "",
        ]

        for label, team in [("TEAM 1", t1), ("TEAM 2", t2)]:
            lines += self._format_team(label, team)

        lines += [
            "=== TECHNICAL NOTES ===",
            "- Coordinates from Homography (ViewTransformer): pitch 68 x 23.32m (camera angle)",
            "- Speed: km/h | Distance: km | Compactness: meters",
            "- Pressing: event of player closing down opponent within 5m, lasting >= 3 frames",
            "- Formation: analyze position snapshot every second",
            "",
            "Please write the tactical analysis report strictly in the specified JSON format.",
        ]

        return "\n".join(lines)

    @staticmethod
    def _format_formation_history(raw: list) -> str:
        """
        formation_history can be:
          - list[str]:  ["4-3-3", "4-4-2"]
          - list[dict]: [{"window_index":0, "formation":"4-3-3", ...}, ...]
          - empty / None
        Returns string "4-3-3 → 4-4-2" or "N/A".
        """
        if not raw:
            return "N/A"
        if isinstance(raw[0], dict):
            parts = [w.get("formation", "?") for w in raw]
        else:
            parts = [str(x) for x in raw]
        return " → ".join(parts) if parts else "N/A"

    def _format_team(self, label: str, team: dict) -> list[str]:
        if not team:
            return [f"  [{label}]: No data\n"]

        zone    = team.get("zone_distribution", {})
        players = team.get("players", {})

        formation_history_str = self._format_formation_history(
            team.get("formation_history", [])
        )

        # Top 3 players by distance
        top3 = sorted(
            players.items(),
            key=lambda x: x[1].get("distance_km", 0),
            reverse=True,
        )[:3]

        # Player with most touches
        top_touch = sorted(
            players.items(),
            key=lambda x: x[1].get("ball_touches", 0),
            reverse=True,
        )[:1]

        lines = [
            "┌─────────────────────────────────────────┐",
            f"│  {label:^39}│",
            "└─────────────────────────────────────────┘",
            f"  Dominant formation  : {team.get('dominant_formation', 'N/A')}",
            f"  Formation history   : {formation_history_str}",
            f"  Possession          : {team.get('possession_pct', 0):.1f}%",
            f"  Total distance      : {team.get('total_distance_km', 0):.3f} km",
            f"  Average speed       : {team.get('avg_speed_kmh', 0):.1f} km/h",
            f"  Max speed           : {team.get('max_speed_kmh', 0):.1f} km/h",
            f"  Avg compactness     : {team.get('avg_compactness_m', 0):.1f} m",
            f"  Pressing events     : {team.get('pressing_events', 0)}",
            f"  Zone distribution   :",
            f"    • Defensive       : {zone.get('defensive', 0):.1f}%",
            f"    • Middle          : {zone.get('middle', 0):.1f}%",
            f"    • Attacking       : {zone.get('attacking', 0):.1f}%",
            f"  Tracked players     : {len(players)}",
        ]

        if top3:
            lines.append("  Top 3 players (distance):")
            for pid, ps in top3:
                lines.append(
                    f"    • #{pid}: {ps.get('distance_km', 0):.3f} km | "
                    f"avg {ps.get('avg_speed_kmh', 0):.1f} km/h | "
                    f"max {ps.get('max_speed_kmh', 0):.1f} km/h | "
                    f"touches {ps.get('ball_touches', 0)}"
                )

        if top_touch and top_touch[0][1].get("ball_touches", 0) > 0:
            pid, ps = top_touch[0]
            lines.append(
                f"  Player with most touches: #{pid} "
                f"({ps.get('ball_touches', 0)} touches)"
            )

        lines.append("")
        return lines