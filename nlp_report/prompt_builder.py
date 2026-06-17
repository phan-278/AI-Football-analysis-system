import json
from typing import Any


class PromptBuilder:
    """
    Chuyển match_stats → (system_prompt, user_prompt).

    Usage:
        builder = PromptBuilder(match_stats)
        system_prompt, user_prompt = builder.build()
    """

    SYSTEM_PROMPT = """Bạn là chuyên gia phân tích chiến thuật bóng đá chuyên nghiệp với hơn 10 năm kinh nghiệm.
Nhiệm vụ: đọc số liệu thống kê trận đấu được trích xuất bằng AI Computer Vision
và viết báo cáo phân tích chiến thuật chuyên sâu bằng tiếng Việt.

Yêu cầu báo cáo:
- Phân tích sơ đồ chiến thuật và phong cách chơi từng đội
- So sánh pressing, kiểm soát bóng, sử dụng không gian sân
- Chỉ ra điểm mạnh và điểm yếu cụ thể dựa trên số liệu
- Ngôn ngữ chuyên nghiệp, dùng thuật ngữ bóng đá chuẩn
- Nhận xét phải bám sát vào con số thực tế được cung cấp

Trả lời DUY NHẤT dưới dạng JSON hợp lệ với cấu trúc sau (không thêm markdown, không giải thích):
{
  "match_overview": "Tổng quan trận đấu 2-3 câu, đề cập đến thời lượng và phong cách chơi tổng thể",
  "team1_analysis": {
    "title": "Tiêu đề ngắn mô tả phong cách đội 1",
    "formation": "Mô tả sơ đồ và cách triển khai chiến thuật",
    "strengths": ["điểm mạnh cụ thể dựa trên số liệu 1", "điểm mạnh 2", "điểm mạnh 3"],
    "weaknesses": ["điểm yếu 1", "điểm yếu 2"],
    "tactical_summary": "Đoạn phân tích chiến thuật chi tiết 4-5 câu, đề cập cụ thể các con số"
  },
  "team2_analysis": {
    "title": "Tiêu đề ngắn mô tả phong cách đội 2",
    "formation": "Mô tả sơ đồ và cách triển khai chiến thuật",
    "strengths": ["điểm mạnh 1", "điểm mạnh 2", "điểm mạnh 3"],
    "weaknesses": ["điểm yếu 1", "điểm yếu 2"],
    "tactical_summary": "Đoạn phân tích chiến thuật chi tiết 4-5 câu, đề cập cụ thể các con số"
  },
  "comparison": {
    "possession_battle": "Nhận xét chi tiết về trận chiến kiểm soát bóng",
    "pressing_duel": "So sánh cường độ và hiệu quả pressing 2 đội dựa trên số liệu",
    "space_usage": "Phân tích cách 2 đội khai thác không gian sân, dựa vào zone distribution",
    "physical_comparison": "So sánh thể lực: tổng quãng đường, tốc độ trung bình, tốc độ tối đa",
    "key_difference": "Sự khác biệt chiến thuật cốt lõi quyết định tính chất trận đấu"
  },
  "key_players": [
    {
      "team": "Đội 1 hoặc Đội 2",
      "player_id": "ID cầu thủ",
      "role": "Vai trò chiến thuật",
      "highlight": "Nhận xét ngắn về cầu thủ này dựa trên số liệu"
    }
  ],
  "conclusion": "Kết luận tổng quan 3-4 câu về chiến thuật và diễn biến trận đấu"
}"""

    def __init__(self, match_stats: dict[str, Any]):
        self.stats = match_stats

    @classmethod
    def from_json_file(cls, path: str) -> "PromptBuilder":
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def build(self) -> tuple[str, str]:
        """Trả về (system_prompt, user_prompt)."""
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
            "=== DỮ LIỆU THỐNG KÊ TRẬN ĐẤU (trích xuất bằng AI Computer Vision) ===",
            "",
            f"Video       : {info.get('video_file', 'N/A')}",
            f"Thời lượng  : {dur_min} phút  |  FPS: {info.get('fps', 24)}  |  "
            f"Tổng frames: {info.get('total_frames', 0)}",
            "",
            "┌─────────────────────────────────────────┐",
            "│           KIỂM SOÁT BÓNG                │",
            "└─────────────────────────────────────────┘",
            f"  Đội 1: {ball.get('possession_team1_pct', 0):.1f}%",
            f"  Đội 2: {ball.get('possession_team2_pct', 0):.1f}%",
            "",
        ]

        for label, team in [("ĐỘI 1", t1), ("ĐỘI 2", t2)]:
            lines += self._format_team(label, team)

        lines += [
            "=== GHI CHÚ KỸ THUẬT ===",
            "- Tọa độ từ Homography (ViewTransformer): sân 68 × 23.32m (góc camera)",
            "- Tốc độ: km/h | Quãng đường: km | Compactness: mét",
            "- Pressing: số sự kiện cầu thủ áp sát đối phương trong vòng 5m, kéo dài ≥ 3 frame",
            "- Formation: phân tích snapshot vị trí mỗi giây",
            "",
            "Hãy viết báo cáo phân tích chiến thuật theo đúng định dạng JSON đã chỉ định.",
        ]

        return "\n".join(lines)

    @staticmethod
    def _format_formation_history(raw: list) -> str:
        """
        formation_history có thể là:
          - list[str]:  ["4-3-3", "4-4-2"]
          - list[dict]: [{"window_index":0, "formation":"4-3-3", ...}, ...]
          - rỗng / None
        Trả về string "4-3-3 → 4-4-2" hoặc "N/A".
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
            return [f"  [{label}]: Không có dữ liệu\n"]

        zone    = team.get("zone_distribution", {})
        players = team.get("players", {})

        formation_history_str = self._format_formation_history(
            team.get("formation_history", [])
        )

        # Top 3 cầu thủ theo quãng đường
        top3 = sorted(
            players.items(),
            key=lambda x: x[1].get("distance_km", 0),
            reverse=True,
        )[:3]

        # Cầu thủ nhiều touches nhất
        top_touch = sorted(
            players.items(),
            key=lambda x: x[1].get("ball_touches", 0),
            reverse=True,
        )[:1]

        lines = [
            "┌─────────────────────────────────────────┐",
            f"│  {label:^39}│",
            "└─────────────────────────────────────────┘",
            f"  Sơ đồ chủ đạo       : {team.get('dominant_formation', 'N/A')}",
            f"  Lịch sử sơ đồ       : {formation_history_str}",
            f"  Kiểm soát bóng      : {team.get('possession_pct', 0):.1f}%",
            f"  Tổng quãng đường    : {team.get('total_distance_km', 0):.3f} km",
            f"  Tốc độ trung bình   : {team.get('avg_speed_kmh', 0):.1f} km/h",
            f"  Tốc độ tối đa       : {team.get('max_speed_kmh', 0):.1f} km/h",
            f"  Độ compact TB       : {team.get('avg_compactness_m', 0):.1f} m",
            f"  Số lần pressing     : {team.get('pressing_events', 0)}",
            f"  Phân bố zone        :",
            f"    • Phòng thủ       : {zone.get('defensive', 0):.1f}%",
            f"    • Giữa sân        : {zone.get('middle', 0):.1f}%",
            f"    • Tấn công        : {zone.get('attacking', 0):.1f}%",
            f"  Số cầu thủ tracking : {len(players)}",
        ]

        if top3:
            lines.append("  Top 3 cầu thủ (quãng đường):")
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
                f"  Cầu thủ nhiều touches nhất: #{pid} "
                f"({ps.get('ball_touches', 0)} touches)"
            )

        lines.append("")
        return lines