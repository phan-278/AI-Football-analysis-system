import os
import json
import argparse
from dotenv import load_dotenv

from nlp_report.llm_client import LLMClient
from nlp_report.pdf_builder import PDFBuilder

def adapt_stats(root_stats: dict) -> dict:
    """Chuyển đổi match_stats.json từ root stats_aggregator sang format của nlp_report."""
    def _adapt_team(team_id: int) -> dict:
        t_key = f"team_{team_id}"
        team_data = root_stats.get("teams", {}).get(t_key, {})
        
        players = root_stats.get("players", [])
        team_players = {}
        max_speeds = []
        zone_counts = {"defensive": 0, "middle": 0, "attacking": 0}
        
        for p in players:
            if p.get("team") == team_id:
                pid = str(p.get("player_id"))
                dist_km = p.get("total_distance_meters", 0) / 1000
                avg_s = p.get("average_speed_kmh", 0)
                max_s = p.get("max_speed_kmh", 0)
                max_speeds.append(max_s)
                
                # Zone distribution từ player
                p_zone = p.get("zone_distribution", {})
                for z, v in p_zone.items():
                    # Map zone names nếu có khác biệt
                    z_lower = z.lower()
                    if "def" in z_lower:
                        zone_counts["defensive"] += v
                    elif "mid" in z_lower:
                        zone_counts["middle"] += v
                    elif "att" in z_lower:
                        zone_counts["attacking"] += v
                        
                team_players[pid] = {
                    "distance_km": dist_km,
                    "avg_speed_kmh": avg_s,
                    "max_speed_kmh": max_s,
                    "ball_touches": p.get("possession_frames", 0),
                    "zone_distribution": p_zone
                }
                
        total_z = sum(zone_counts.values()) or 1
        team_zone = {k: round(v / total_z * 100, 1) for k, v in zone_counts.items()}
                
        return {
            "possession_pct": team_data.get("possession_percentage", 0),
            "total_distance_km": team_data.get("total_distance_meters", 0) / 1000,
            "avg_speed_kmh": team_data.get("average_speed_kmh", 0),
            "max_speed_kmh": max(max_speeds) if max_speeds else 0,
            "dominant_formation": team_data.get("overall_formation", "N/A"),
            "formation_history": team_data.get("formation_history", []),
            "avg_compactness_m": team_data.get("compactness", {}).get("average_spread_meters", 0),
            "pressing_events": int(team_data.get("pressing", {}).get("pressing_intensity_percentage", 0)),
            "zone_distribution": team_zone,
            "players": team_players
        }

    nlp_stats = {
        "match_info": {
            "duration_seconds": root_stats.get("match_summary", {}).get("match_duration_seconds", 0),
            "fps": root_stats.get("match_summary", {}).get("frame_rate", 24),
            "total_frames": root_stats.get("match_summary", {}).get("total_frames", 0),
            "video_file": "Video (Từ Stats Exporter)"
        },
        "ball": {
            "possession_team1_pct": root_stats.get("teams", {}).get("team_1", {}).get("possession_percentage", 50),
            "possession_team2_pct": root_stats.get("teams", {}).get("team_2", {}).get("possession_percentage", 50)
        },
        "team1": _adapt_team(1),
        "team2": _adapt_team(2)
    }
    return nlp_stats

def run_report(match_stats_path: str, output_pdf_path: str, provider: str = "claude"):
    print("[1/3] Đang tải biến môi trường (API Key)...")
    load_dotenv()
    
    if not os.path.exists(match_stats_path):
        raise FileNotFoundError(f"Không tìm thấy file {match_stats_path}. Bạn cần chạy 'python main.py' trước để tạo file này.")
    
    print(f"[2/3] Đọc dữ liệu từ {match_stats_path} và gọi API ({provider})...")
    with open(match_stats_path, 'r', encoding='utf-8') as f:
        root_stats = json.load(f)
        
    # CHUYỂN ĐỔI STATS CỦA BẠN THÀNH STATS MÀ NLP YÊU CẦU
    adapted_stats = adapt_stats(root_stats)
        
    try:
        # Khởi tạo LLM Client (sẽ tự lấy API Key từ biến môi trường ANTHROPIC_API_KEY hoặc OPENAI_API_KEY)
        client = LLMClient(provider=provider)
        report_json = client.generate_from_stats(adapted_stats)
        print("      -> API trả lời thành công.")
    except Exception as e:
        print(f"\n[LỖI API] Lỗi khi gọi LLM: {e}")
        print("Vui lòng kiểm tra lại API Key trong file .env hoặc kết nối mạng.")
        return

    print(f"[3/3] Đang tạo file PDF báo cáo...")
    pdf_builder = PDFBuilder(
        report=report_json,
        match_stats=adapted_stats,  # Dùng adapted_stats cho PDF luôn
        team1_name="Đội 1 (Xanh)",
        team2_name="Đội 2 (Đỏ)"
    )
    saved_path = pdf_builder.save(output_pdf_path)
    print(f"✅ HOÀN TẤT! Báo cáo đã được lưu tại: {saved_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sinh báo cáo chiến thuật (PDF) bằng NLP từ match_stats.json.")
    # Sửa default path thành thư mục outputs/ nơi mà root StatsExporter lưu file
    parser.add_argument("--stats_path", type=str, default="outputs/match_stats.json", help="Đường dẫn đến file JSON kết quả tracking.")
    parser.add_argument("--output", type=str, default="output_videos/tactical_report.pdf", help="Đường dẫn lưu file PDF đầu ra.")
    parser.add_argument("--provider", type=str, choices=["gemini", "claude", "openai","groq"], default="gemini", help="Model LLM (gemini, claude hoặc openai).")
    
    args = parser.parse_args()
    
    run_report(
        match_stats_path=args.stats_path,
        output_pdf_path=args.output,
        provider=args.provider
    )
