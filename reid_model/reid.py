"""--------------------------------------------------
Giữ nguyên pattern: track → reid → merge_tracks_offline()
Thêm 3 kỹ thuật football-specific:
  1. Team color filter  (lọc gallery cùng team trước khi embed)
  2. Spatial constraint vật lý  (max tốc độ 11 m/s)
  3. Formation prior  (tie-breaker theo primary_zone)

Thay đổi so với studio version:
  - jacket_dist_thresh: 30px → 60px  (cầu thủ di chuyển nhanh hơn)
  - last_pos: pixel → meters  (dùng position_transformed)
  - gallery thêm 'team' và 'primary_zone'
  - adaptive threshold: 3 mức theo time_gap thay vì 2
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
import torchreid
import os
import pickle


from utils import get_foot_position, measure_distance, get_cosine_similarity


# ──────────────────────────────────────────────
#  Hằng số vật lý sân bóng
# ──────────────────────────────────────────────
MAX_PLAYER_SPEED_MS  = 11.0   # m/s  (~40 km/h, sprint tối đa)
FRAME_RATE           = 24     # fps

# Zone mapping: chia sân 105×68m thành grid 3×2
ZONE_COLS = [0, 35, 70, 105]   # left / center / right
ZONE_ROWS = [0, 34, 68]        # defense / attack (theo chiều dọc)
ZONE_NAMES = {
    (0, 0): "left_def",   (1, 0): "center_def",   (2, 0): "right_def",
    (0, 1): "left_att",   (1, 1): "center_att",   (2, 1): "right_att",
}


def _get_zone(pos_meters) -> str:
    """Chuyển tọa độ thực (x, y) → tên zone."""
    if pos_meters is None:
        return "unknown"
    x, y = pos_meters
    col = min(2, sum(1 for b in ZONE_COLS[1:-1] if x >= b))
    row = min(1, sum(1 for b in ZONE_ROWS[1:-1] if y >= b))
    return ZONE_NAMES.get((col, row), "unknown")


class ReID:
    """
    Football ReID — pattern giống studio reid.py.

    Sử dụng:
        reid = ReID(device='cpu')
        merged = reid.merge_tracks_offline(frames, tracks)
        # merged["players"] là tracks dict với global_id ổn định
    """

    def __init__(
        self,
        model_name           = 'osnet_x1_0',
        similarity_threshold = 0.75,          # base threshold (time_gap ngắn)
        jacket_dist_thresh   = 30,            # pixel — lên 60 (football)
        jacket_time_thresh   = 24,            # frame — 2 giây ở 24fps
        long_term_thresh     = 18000,         # frame — ~12 phút (không dùng, giữ tương thích)
        device               = 'cpu',
    ):
        self.similarity_threshold = similarity_threshold
        self.jacket_dist_thresh   = jacket_dist_thresh
        self.jacket_time_thresh   = jacket_time_thresh
        self.long_term_thresh     = long_term_thresh

        self.device = device if not (device == 'cuda' and not torch.cuda.is_available()) else 'cpu'

        # Load OSNet x1_0
        self.model = torchreid.models.build_model(
            name=model_name, num_classes=1000, pretrained=True
        ).to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # Gallery: {global_id: {'features', 'last_pos', 'last_frame', 'team', 'zone_counts'}}
        self.gallery: dict = {}

    # ──────────────────────────────────────────
    #  Extract feature  (giữ nguyên từ studio)
    # ──────────────────────────────────────────
    def extract_feature(self, frame: np.ndarray, bbox: list):
        x1, y1, x2, y2 = map(int, bbox)
        if x2 - x1 < 20 or y2 - y1 < 40:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model(tensor)
            feat = F.normalize(feat, p=2, dim=1)
        return feat.cpu().numpy().flatten()

    # ──────────────────────────────────────────
    #  Match với gallery  (5 lớp kỹ thuật)
    # ──────────────────────────────────────────
    def _match_with_gallery(
        self,
        current_feat,
        current_pos_px,       # pixel (foot position)
        current_pos_m,        # meters (position_transformed), có thể None
        current_frame: int,
        current_team: int,
    ):
        """
        5 lớp theo thứ tự ưu tiên:
          1. Jacket logic       — vị trí gần + thời gian ngắn
          2. Team color filter  — chỉ xét cùng team
          3. Spatial constraint — loại vi phạm vật lý
          4. OSNet embedding    — adaptive threshold
          5. Formation prior    — tie-breaker zone
        Returns (best_id, score) hoặc (None, 0)
        """
        best_id    = None
        best_score = -1.0
        candidates = []   # (score, id) để formation prior tie-break

        for gid, data in self.gallery.items():
            time_gap = current_frame - data['last_frame']

            # ── Lớp 1: Jacket logic (giữ nguyên từ studio) ──────────────
            dist_px = measure_distance(current_pos_px, data['last_pos_px'])
            if dist_px < self.jacket_dist_thresh and time_gap < self.jacket_time_thresh:
                return gid, 1.0

            # ── Lớp 2: Team color filter (football-specific) ─────────────
            if data.get('team') is not None and data['team'] != current_team:
                continue   # khác team → bỏ qua hoàn toàn

            # ── Lớp 3: Spatial constraint vật lý ─────────────────────────
            if current_pos_m is not None and data.get('last_pos_m') is not None:
                elapsed_sec     = time_gap / FRAME_RATE
                max_dist_meters = MAX_PLAYER_SPEED_MS * elapsed_sec
                actual_dist_m   = measure_distance(current_pos_m, data['last_pos_m'])
                if actual_dist_m > max_dist_meters + 2.0:   # +2m buffer GPS error
                    continue   # vật lý bất khả thi → bỏ qua

            # ── Lớp 4: OSNet embedding + adaptive threshold ───────────────
            if current_feat is None or not data['features']:
                continue

            sims = [get_cosine_similarity(current_feat, f) for f in data['features']]
            score = max(sims)

            # Adaptive threshold theo time_gap
            if time_gap < 72:        # < 3 giây
                thresh = self.similarity_threshold          # 0.75
            elif time_gap < 720:     # 3 – 30 giây
                thresh = self.similarity_threshold + 0.05  # 0.80
            else:                    # > 30 giây (long-term)
                thresh = self.similarity_threshold + 0.10  # 0.85

            if score >= thresh:
                candidates.append((score, gid))

        if not candidates:
            return None, 0.0

        # ── Lớp 5: Formation prior — tie-breaker ─────────────────────────
        if len(candidates) == 1:
            best_score, best_id = candidates[0]
        else:
            # Nếu 2 candidate có score gần nhau (< 0.03), dùng zone để phân giải
            candidates.sort(reverse=True)
            top_score, top_id = candidates[0]
            second_score, second_id = candidates[1]

            if top_score - second_score < 0.03:
                # Tie: chọn người có primary_zone khớp vị trí hiện tại hơn
                current_zone = _get_zone(current_pos_m)
                top_zone    = self.gallery[top_id].get('primary_zone', 'unknown')
                second_zone = self.gallery[second_id].get('primary_zone', 'unknown')

                if second_zone == current_zone and top_zone != current_zone:
                    best_id, best_score = second_id, second_score
                else:
                    best_id, best_score = top_id, top_score
            else:
                best_id, best_score = top_id, top_score

        return best_id, best_score

    # ──────────────────────────────────────────
    #  Gallery helpers
    # ──────────────────────────────────────────
    def _update_gallery(self, gid, feat, pos_px, pos_m, frame_idx, team):
        if gid not in self.gallery:
            self.gallery[gid] = {
                'features'     : [],
                'last_pos_px'  : pos_px,
                'last_pos_m'   : pos_m,
                'last_frame'   : frame_idx,
                'team'         : team,
                'zone_counts'  : {},      # {zone_name: count}
                'primary_zone' : 'unknown',
            }

        data = self.gallery[gid]

        # Rolling buffer 10 embeddings (giữ nguyên từ studio)
        if feat is not None and len(data['features']) < 10:
            data['features'].append(feat)

        data['last_pos_px']  = pos_px
        data['last_pos_m']   = pos_m
        data['last_frame']   = frame_idx
        if team is not None:
            data['team'] = team

        # Cập nhật zone_counts → primary_zone
        if pos_m is not None:
            zone = _get_zone(pos_m)
            data['zone_counts'][zone] = data['zone_counts'].get(zone, 0) + 1
            data['primary_zone'] = max(data['zone_counts'], key=data['zone_counts'].get)

    # ──────────────────────────────────────────
    #  Offline merge  (giữ pattern từ studio)
    # ──────────────────────────────────────────
    def merge_tracks_offline(self, frames: list, tracks: dict) -> dict:
        """
        Xử lý gộp ID offline — cùng pattern với studio reid.py.

        Input:
            frames: list BGR frames
            tracks: dict với tracks["players"][frame_num][track_id] = {
                        "bbox": [...],
                        "position":             (x_px, y_px),
                        "position_transformed": (x_m, y_m) hoặc None,
                        "team":                 int hoặc None,
                    }

        Output:
            tracks dict với track_id đã được thay bằng global_id ổn định
        """
        player_tracks     = tracks.get("players", [])
        new_player_tracks = [{} for _ in range(len(frames))]
        id_map            = {}   # bytetrack_id → global_id (giống studio)

        for frame_idx, frame_data in enumerate(player_tracks):
            frame = frames[frame_idx]

            for old_id, track_info in frame_data.items():
                bbox    = track_info['bbox']
                pos_px  = track_info.get('position') or get_foot_position(bbox)
                pos_m   = track_info.get('position_transformed')   # None nếu ngoài H
                team    = track_info.get('team')

                feat = self.extract_feature(frame, bbox)

                # Tra cứu id_map (giống studio)
                actual_id = id_map.get(old_id)

                if actual_id is None:
                    matched_id, sim = self._match_with_gallery(
                        feat, pos_px, pos_m, frame_idx, team
                    )
                    if matched_id is not None:
                        actual_id       = matched_id
                        id_map[old_id]  = actual_id
                        if frame_idx % 200 == 0:
                            print(f"[ReID] Frame {frame_idx}: "
                                  f"track #{old_id} → recovered global #{actual_id} "
                                  f"(sim={sim:.3f})")
                    else:
                        actual_id      = old_id
                        id_map[old_id] = actual_id

                # Cập nhật gallery
                self._update_gallery(actual_id, feat, pos_px, pos_m, frame_idx, team)

                # Ghi kết quả — giữ toàn bộ track_info, chỉ đổi key
                new_player_tracks[frame_idx][actual_id] = track_info

            # Log tiến độ
            if frame_idx % 500 == 0:
                print(f"[ReID] Progress {frame_idx}/{len(player_tracks)} frames "
                      f"| gallery size: {len(self.gallery)} IDs")

        # Giữ referees và ball không thay đổi
        result = dict(tracks)
        result["players"] = new_player_tracks
        return result

    # ──────────────────────────────────────────
    #  Debug helper
    # ──────────────────────────────────────────
    def gallery_summary(self) -> dict:
        return {
            gid: {
                "team"         : d['team'],
                "primary_zone" : d['primary_zone'],
                "num_features" : len(d['features']),
                "last_frame"   : d['last_frame'],
            }
            for gid, d in self.gallery.items()
        }
    # ──────────────────────────────────────────
    #  Save to stubs
    # ──────────────────────────────────────────
    def save_gallery(self, stub_path):
        """Lưu toàn bộ dữ liệu gallery (features, zones, team) vào file stub."""
        os.makedirs(os.path.dirname(stub_path), exist_ok=True)
        with open(stub_path, 'wb') as f:
            pickle.dump(self.gallery, f)
        print(f"[ReID] Gallery saved to {stub_path}")

    def load_gallery(self, stub_path):
        """Tải dữ liệu gallery từ file stub nếu tồn tại."""
        if os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                self.gallery = pickle.load(f)
            print(f"[ReID] Gallery loaded from {stub_path}")
            return True
        return False

    def save_merged_tracks(self, tracks, stub_path):
        """Lưu kết quả tracks sau khi đã ReID."""
        os.makedirs(os.path.dirname(stub_path), exist_ok=True)
        with open(stub_path, 'wb') as f:
            pickle.dump(tracks, f)
        print(f"[ReID] Merged tracks saved to {stub_path}")