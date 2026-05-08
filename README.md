# AI-Football-analysis-system
football_analysis/
│
├── main.py
│   # Orchestrate toàn bộ pipeline mới
│
├── yolo_inference.py
├── README.md
├── .gitignore
│
├── trackers/
│   ├── __init__.py
│   └── tracker.py
│       # ByteTrack → StrongSORT + OSNet
│
├── homography/              ★ module mới
│   ├── __init__.py
│   ├── keypoint_detector.py
│   │   # Detect vạch sân → 2D keypoints
│   ├── homography_estimator.py
│   │   # RANSAC → H matrix / frame
│   └── pitch_template.py
│       # Sân chuẩn 105×68m, keypoint reference
│
├── view_transformer/
│   ├── __init__.py
│   └── view_transformer.py
│       # Nhận H động từ homography/
│
├── eagle_view/              ★ module mới
│   ├── __init__.py
│   ├── pitch_renderer.py
│   │   # Vẽ mini-pitch nền (SVG → numpy)
│   └── eagle_view_renderer.py
│       # Chấm màu + trail + PiP ghép video
│
├── speed_and_distance_estimator/
│   ├── __init__.py
│   └── speed_and_distance_estimator.py
│       # Dùng H động (không cố định)
│
├── stats_aggregator/        ★ module mới
│   ├── __init__.py
│   ├── player_stats.py
│   │   # Distance, speed, zone, possession
│   ├── team_stats.py
│   │   # Compactness, pressing, avg speed
│   ├── formation_detector.py
│   │   # K-means → "4-3-3" mỗi 30 giây
│   ├── heatmap_generator.py
│   │   # Gaussian heatmap → PNG
│   └── exporter.py
│       # Gom tất cả → match_stats.json
│
├── nlp_report/              ★ module mới (người khác viết)
│   ├── __init__.py
│   ├── prompt_builder.py
│   │   # JSON stats → LLM prompt
│   ├── llm_client.py
│   │   # Gọi Claude / GPT API
│   └── pdf_builder.py
│       # ReportLab → PDF chiến thuật
│
├── camera_movement_estimator/   # Giữ nguyên
├── team_assigner/               # Giữ nguyên
├── player_ball_assigner/        # Giữ nguyên
│
├── utils/
│   └── math_utils.py            # Thêm mới
│
├── models/
│   ├── best_670img_yolo11s.pt
│   └── osnet_x0_25_market.pt
│       # Re-ID backbone (~2MB)
│
├── input_videos/                # Giữ nguyên
├── output_videos/               # Giữ nguyên
│
├── stubs/
│   ├── track_stubs.pkl
│   ├── camera_movement.pkl
│   └── homography_stubs.pkl
│       # Cache H matrix toàn video
│
└── outputs/                 ★ thư mục mới
    ├── heatmaps/
    │   ├── player_1.png
    │   └── team1_combined.png
    │
    ├── match_stats.json     # Data contract → NLP
    └── tactical_report.pdf


tracks["players"][frame_num][track_id] = {
    "bbox": [x1, y1, x2, y2],

    "position": (x, y),                  # pixel (foot position)
    "position_adjusted": (x, y),         # sau khi trừ camera movement
    "position_transformed": (x, y),      # tọa độ thật (meters)

    "speed": float,                      # km/h
    "distance": float,                   # tổng quãng đường

    "team": int,                         # 1 hoặc 2
    "team_color": (B, G, R),             # màu vẽ

    "has_ball": bool                     # có đang giữ bóng không
}