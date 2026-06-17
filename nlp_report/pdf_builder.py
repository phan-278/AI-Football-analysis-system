import os
from datetime import datetime
from typing import Any

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable,
        KeepTogether,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    REPORTLAB_AVAILABLE = True
except ImportError as e:
    raise ImportError("\n\n❌ LỖI: Bạn chưa cài đặt thư viện 'reportlab' để xuất file PDF.\n👉 Hãy chạy lệnh này trong Terminal: pip install reportlab\n") from e


# ─── Màu sắc ────────────────────────────────────────────────────────────────
DARK        = colors.HexColor("#0f172a")
DARK2       = colors.HexColor("#1e293b")
TEAM1_C     = colors.HexColor("#2563eb")   # xanh dương
TEAM2_C     = colors.HexColor("#dc2626")   # đỏ
GOLD        = colors.HexColor("#f59e0b")   # vàng
LIGHT       = colors.HexColor("#f8fafc")
BORDER      = colors.HexColor("#e2e8f0")
GRAY        = colors.HexColor("#94a3b8")
TEXT        = colors.HexColor("#1e293b")
WHITE       = colors.white
GREEN_LIGHT = colors.HexColor("#f0fdf4")
GREEN_DARK  = colors.HexColor("#16a34a")
RED_LIGHT   = colors.HexColor("#fef2f2")
RED_DARK    = colors.HexColor("#dc2626")
# ────────────────────────────────────────────────────────────────────────────


class PDFBuilder:
    """
    Build PDF báo cáo chiến thuật.

    Parameters
    ----------
    report      : dict  — output của LLMClient.generate()
    match_stats : dict  — output của StatsAggregator.compute()
    team1_name  : str   — tên hiển thị đội 1 (mặc định "Đội 1")
    team2_name  : str   — tên hiển thị đội 2 (mặc định "Đội 2")
    """

    def __init__(
        self,
        report: dict[str, Any],
        match_stats: dict[str, Any],
        team1_name: str = "Đội 1",
        team2_name: str = "Đội 2",
    ):
        if not REPORTLAB_AVAILABLE:
            raise ImportError("Cần cài: pip install reportlab")

        self.report     = report
        self.stats      = match_stats
        self.t1_name    = team1_name
        self.t2_name    = team2_name
        self.S          = self._make_styles()

    # ──────────────────────────────────────────────
    # Public
    # ──────────────────────────────────────────────

    def save(self, output_path: str) -> str:
        """Build PDF và lưu. Trả về đường dẫn tuyệt đối."""
        os.makedirs(
            os.path.dirname(os.path.abspath(output_path)),
            exist_ok=True,
        )
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=1.8 * cm,
            leftMargin=1.8 * cm,
            topMargin=2 * cm,
            bottomMargin=1.8 * cm,
        )
        story = (
            self._s_header()
            + self._s_overview()
            + self._s_quick_compare()
            + self._s_team("team1")
            + self._s_team("team2")
            + self._s_comparison_table()
            + self._s_comparison_text()
            + self._s_key_players()
            + self._s_conclusion()
            + self._s_footer()
        )
        doc.build(story)
        return os.path.abspath(output_path)

    # ──────────────────────────────────────────────
    # Styles
    # ──────────────────────────────────────────────

    def _make_styles(self) -> dict:
        return {
            "h_title": ParagraphStyle(
                "h_title", fontName="Helvetica-Bold", fontSize=20,
                textColor=WHITE, alignment=TA_CENTER, spaceAfter=2,
            ),
            "h_vs": ParagraphStyle(
                "h_vs", fontName="Helvetica-Bold", fontSize=13,
                textColor=GOLD, alignment=TA_CENTER, spaceAfter=0,
            ),
            "h_sub": ParagraphStyle(
                "h_sub", fontName="Helvetica", fontSize=9,
                textColor=GRAY, alignment=TA_CENTER,
            ),
            "sec": ParagraphStyle(
                "sec", fontName="Helvetica-Bold", fontSize=12,
                textColor=WHITE, spaceBefore=10, spaceAfter=5,
            ),
            "body": ParagraphStyle(
                "body", fontName="Helvetica", fontSize=10,
                textColor=TEXT, alignment=TA_JUSTIFY, leading=16, spaceAfter=5,
            ),
            "bullet": ParagraphStyle(
                "bullet", fontName="Helvetica", fontSize=9,
                textColor=TEXT, leftIndent=10, spaceAfter=2, leading=14,
            ),
            "center_bold": ParagraphStyle(
                "center_bold", fontName="Helvetica-Bold", fontSize=9,
                textColor=TEXT, alignment=TA_CENTER,
            ),
            "small_gray": ParagraphStyle(
                "small_gray", fontName="Helvetica", fontSize=8,
                textColor=GRAY, alignment=TA_CENTER,
            ),
            "footer": ParagraphStyle(
                "footer", fontName="Helvetica", fontSize=8,
                textColor=GRAY, alignment=TA_CENTER,
            ),
            "comp_label": ParagraphStyle(
                "comp_label", fontName="Helvetica-Bold", fontSize=9,
                textColor=colors.HexColor("#475569"),
            ),
        }

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    @property
    def _W(self) -> float:
        """Chiều rộng nội dung (17cm)."""
        return 17 * cm

    def _sec_bar(self, title: str, accent=GOLD) -> Table:
        """Dark section header bar."""
        t = Table(
            [[Paragraph(title, self.S["sec"])]],
            colWidths=[self._W],
        )
        t.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, -1), DARK),
            ("LEFTPADDING",  (0, 0), (-1, -1), 12),
            ("TOPPADDING",   (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 7),
            ("LINEBELOW",    (0, -1), (-1, -1), 2, accent),
        ]))
        return t

    def _stat_card_row(self, cards: list[tuple[str, str]], accent) -> Table:
        """Hàng stat cards."""
        n = len(cards)
        w = self._W / n
        labels = [Paragraph(lbl, self.S["small_gray"]) for lbl, _ in cards]
        values = [
            Paragraph(val, ParagraphStyle(
                "cv", fontName="Helvetica-Bold", fontSize=13,
                textColor=TEXT, alignment=TA_CENTER,
            ))
            for _, val in cards
        ]
        t = Table([labels, values], colWidths=[w] * n)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), LIGHT),
            ("LINEABOVE",     (0, 0), (-1, 0), 2.5, accent),
            ("GRID",          (0, 0), (-1, -1), 0.3, BORDER),
            ("TOPPADDING",    (0, 0), (-1, 0), 7),
            ("BOTTOMPADDING", (0, -1),(-1, -1), 7),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return t

    def _sw_table(self, strengths: list, weaknesses: list) -> Table:
        """Bảng Strengths / Weaknesses 2 cột."""
        max_r = max(len(strengths), len(weaknesses))
        s_pad = strengths + [""] * (max_r - len(strengths))
        w_pad = weaknesses + [""] * (max_r - len(weaknesses))

        header = [
            Paragraph("✅  Điểm mạnh", ParagraphStyle(
                "sh", fontName="Helvetica-Bold", fontSize=9,
                textColor=WHITE, alignment=TA_CENTER,
            )),
            Paragraph("⚠️  Điểm yếu", ParagraphStyle(
                "wh", fontName="Helvetica-Bold", fontSize=9,
                textColor=WHITE, alignment=TA_CENTER,
            )),
        ]
        rows = [header] + [
            [
                Paragraph(f"• {s}" if s else "", self.S["bullet"]),
                Paragraph(f"• {w}" if w else "", self.S["bullet"]),
            ]
            for s, w in zip(s_pad, w_pad)
        ]
        hw = self._W / 2
        t = Table(rows, colWidths=[hw, hw])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (0, 0), GREEN_DARK),
            ("BACKGROUND",    (1, 0), (1, 0), RED_DARK),
            ("BACKGROUND",    (0, 1), (0, -1), GREEN_LIGHT),
            ("BACKGROUND",    (1, 1), (1, -1), RED_LIGHT),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("GRID",          (0, 0), (-1, -1), 0.3, BORDER),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ]))
        return t

    # ──────────────────────────────────────────────
    # Sections
    # ──────────────────────────────────────────────

    def _s_header(self) -> list:
        info    = self.stats.get("match_info", {})
        dur_min = round(info.get("duration_seconds", 0) / 60, 1)
        date    = datetime.now().strftime("%d/%m/%Y %H:%M")
        video   = info.get("video_file", "N/A")
        fps     = info.get("fps", 24)

        data = [
            [Paragraph("⚽  BÁO CÁO PHÂN TÍCH CHIẾN THUẬT", self.S["h_title"])],
            [Paragraph(f"{self.t1_name}  ⚔  {self.t2_name}", self.S["h_vs"])],
            [Paragraph(
                f"Video: {video}  |  Thời lượng: {dur_min} phút  |  "
                f"{fps} FPS  |  {date}",
                self.S["h_sub"],
            )],
        ]
        t = Table(data, colWidths=[self._W])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), DARK),
            ("TOPPADDING",    (0, 0), (0, 0), 20),
            ("BOTTOMPADDING", (0, -1),(-1, -1), 18),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("LINEBELOW",     (0, -1),(-1, -1), 2, GOLD),
        ]))
        return [t, Spacer(1, 0.4 * cm)]

    def _s_overview(self) -> list:
        text = self.report.get("match_overview", "")
        return [
            self._sec_bar("📋  TỔNG QUAN TRẬN ĐẤU"),
            Spacer(1, 0.2 * cm),
            Paragraph(text, self.S["body"]),
            Spacer(1, 0.3 * cm),
        ]

    def _s_quick_compare(self) -> list:
        """2 hàng stat cards so sánh nhanh 2 đội."""
        t1 = self.stats.get("team1", {})
        t2 = self.stats.get("team2", {})
        ball = self.stats.get("ball", {})

        def row(items, accent):
            return self._stat_card_row(items, accent)

        # Row 1: team1 cards
        r1_cards = [
            (f"{self.t1_name}", ""),
            ("Kiểm soát bóng", f"{ball.get('possession_team1_pct', 0):.0f}%"),
            ("Quãng đường",    f"{t1.get('total_distance_km', 0):.2f} km"),
            ("Tốc độ TB",      f"{t1.get('avg_speed_kmh', 0):.1f} km/h"),
            ("Pressing",       str(t1.get("pressing_events", 0))),
            ("Compact",        f"{t1.get('avg_compactness_m', 0):.1f} m"),
        ]
        r2_cards = [
            (f"{self.t2_name}", ""),
            ("Kiểm soát bóng", f"{ball.get('possession_team2_pct', 0):.0f}%"),
            ("Quãng đường",    f"{t2.get('total_distance_km', 0):.2f} km"),
            ("Tốc độ TB",      f"{t2.get('avg_speed_kmh', 0):.1f} km/h"),
            ("Pressing",       str(t2.get("pressing_events", 0))),
            ("Compact",        f"{t2.get('avg_compactness_m', 0):.1f} m"),
        ]

        # Override ô đầu tiên (tên đội) thành label
        n = len(r1_cards)
        w = self._W / n

        def _make_name_card(name, color):
            d = [[Paragraph(name, ParagraphStyle(
                "nc", fontName="Helvetica-Bold", fontSize=11,
                textColor=WHITE, alignment=TA_CENTER,
            ))]]
            t = Table(d, colWidths=[w])
            t.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, -1), color),
                ("TOPPADDING",    (0, 0), (-1, -1), 13),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 13),
                ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ]))
            return t

        team1_card = _make_name_card(self.t1_name, TEAM1_C)
        team2_card = _make_name_card(self.t2_name, TEAM2_C)

        # Build combined row tables (skip index 0 — tên)
        row1 = self._stat_card_row(r1_cards[1:], TEAM1_C)
        row2 = self._stat_card_row(r2_cards[1:], TEAM2_C)

        return [
            self._sec_bar("📊  SO SÁNH NHANH 2 ĐỘI"),
            Spacer(1, 0.2 * cm),
            Table([[team1_card, row1]], colWidths=[w, self._W - w]),
            Spacer(1, 0.15 * cm),
            Table([[team2_card, row2]], colWidths=[w, self._W - w]),
            Spacer(1, 0.4 * cm),
        ]

    def _s_team(self, team_key: str) -> list:
        """Section phân tích 1 đội."""
        is_t1    = (team_key == "team1")
        name     = self.t1_name if is_t1 else self.t2_name
        color    = TEAM1_C if is_t1 else TEAM2_C
        analysis = self.report.get(f"{team_key}_analysis", {})
        t_stats  = self.stats.get(team_key, {})
        ball     = self.stats.get("ball", {})
        poss_key = "possession_team1_pct" if is_t1 else "possession_team2_pct"
        poss     = ball.get(poss_key, t_stats.get("possession_pct", 0))

        # Stat cards
        cards = [
            ("Sơ đồ",       t_stats.get("dominant_formation", "N/A")),
            ("Kiểm soát",   f"{poss:.0f}%"),
            ("Quãng đường", f"{t_stats.get('total_distance_km', 0):.2f} km"),
            ("Tốc độ TB",   f"{t_stats.get('avg_speed_kmh', 0):.1f} km/h"),
            ("Tốc độ Max",  f"{t_stats.get('max_speed_kmh', 0):.1f} km/h"),
            ("Pressing",    str(t_stats.get("pressing_events", 0))),
            ("Compact",     f"{t_stats.get('avg_compactness_m', 0):.1f} m"),
        ]

        # Zone bar
        zone     = t_stats.get("zone_distribution", {})
        zone_bar = self._zone_bar(zone, color)

        # Tactical summary
        title    = analysis.get("title", "")
        summary  = analysis.get("tactical_summary", "")
        form_txt = analysis.get("formation", "")

        # Strengths / weaknesses
        sw = self._sw_table(
            analysis.get("strengths", []),
            analysis.get("weaknesses", []),
        )

        icon = "🔵" if is_t1 else "🔴"
        return [
            self._sec_bar(f"{icon}  PHÂN TÍCH: {name.upper()}", accent=color),
            Spacer(1, 0.2 * cm),
            self._stat_card_row(cards, color),
            Spacer(1, 0.2 * cm),
            zone_bar,
            Spacer(1, 0.25 * cm),
            Paragraph(f"<b>{title}</b>", self.S["body"]) if title else Spacer(1, 0),
            Paragraph(f"<i>Sơ đồ & chiến thuật:</i> {form_txt}", self.S["body"]) if form_txt else Spacer(1, 0),
            Paragraph(summary, self.S["body"]) if summary else Spacer(1, 0),
            Spacer(1, 0.15 * cm),
            sw,
            Spacer(1, 0.5 * cm),
        ]

    def _zone_bar(self, zone: dict, color) -> Table:
        """Thanh phân bố zone 3 phần."""
        def_pct = zone.get("defensive", 0)
        mid_pct = zone.get("middle", 0)
        att_pct = zone.get("attacking", 0)

        W = self._W
        cells = [
            Paragraph(
                f"Phòng thủ<br/><b>{def_pct:.0f}%</b>",
                ParagraphStyle("zd", fontName="Helvetica", fontSize=8,
                               textColor=colors.HexColor("#1d4ed8"), alignment=TA_CENTER),
            ),
            Paragraph(
                f"Giữa sân<br/><b>{mid_pct:.0f}%</b>",
                ParagraphStyle("zm", fontName="Helvetica", fontSize=8,
                               textColor=colors.HexColor("#0369a1"), alignment=TA_CENTER),
            ),
            Paragraph(
                f"Tấn công<br/><b>{att_pct:.0f}%</b>",
                ParagraphStyle("za", fontName="Helvetica", fontSize=8,
                               textColor=colors.HexColor("#1e40af"), alignment=TA_CENTER),
            ),
        ]
        t = Table([cells], colWidths=[W * def_pct / 100, W * mid_pct / 100, W * att_pct / 100])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (0, 0), colors.HexColor("#dbeafe")),
            ("BACKGROUND",    (1, 0), (1, 0), colors.HexColor("#bfdbfe")),
            ("BACKGROUND",    (2, 0), (2, 0), colors.HexColor("#93c5fd")),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("GRID",          (0, 0), (-1, -1), 0.3, WHITE),
        ]))
        return t

    def _s_comparison_table(self) -> list:
        """Bảng số liệu chi tiết 2 đội."""
        t1   = self.stats.get("team1", {})
        t2   = self.stats.get("team2", {})
        ball = self.stats.get("ball", {})
        z1   = t1.get("zone_distribution", {})
        z2   = t2.get("zone_distribution", {})

        def _winner(v1, v2, higher_is_better=True):
            """Trả về (bold_t1, bold_t2)."""
            try:
                n1 = float(str(v1).replace("%", "").replace("km", "").replace("km/h", "").strip())
                n2 = float(str(v2).replace("%", "").replace("km", "").replace("km/h", "").strip())
                if higher_is_better:
                    return n1 > n2, n2 > n1
                else:
                    return n1 < n2, n2 < n1
            except Exception:
                return False, False

        rows_data = [
            ("Chỉ số", self.t1_name, self.t2_name),
            ("Sơ đồ chủ đạo",
             t1.get("dominant_formation", "N/A"),
             t2.get("dominant_formation", "N/A")),
            ("Kiểm soát bóng",
             f"{ball.get('possession_team1_pct', 0):.1f}%",
             f"{ball.get('possession_team2_pct', 0):.1f}%"),
            ("Tổng quãng đường",
             f"{t1.get('total_distance_km', 0):.3f} km",
             f"{t2.get('total_distance_km', 0):.3f} km"),
            ("Tốc độ trung bình",
             f"{t1.get('avg_speed_kmh', 0):.1f} km/h",
             f"{t2.get('avg_speed_kmh', 0):.1f} km/h"),
            ("Tốc độ tối đa",
             f"{t1.get('max_speed_kmh', 0):.1f} km/h",
             f"{t2.get('max_speed_kmh', 0):.1f} km/h"),
            ("Số lần pressing",
             str(t1.get("pressing_events", 0)),
             str(t2.get("pressing_events", 0))),
            ("Độ compact (TB)",
             f"{t1.get('avg_compactness_m', 0):.1f} m",
             f"{t2.get('avg_compactness_m', 0):.1f} m"),
            ("Zone phòng thủ",
             f"{z1.get('defensive', 0):.1f}%",
             f"{z2.get('defensive', 0):.1f}%"),
            ("Zone giữa sân",
             f"{z1.get('middle', 0):.1f}%",
             f"{z2.get('middle', 0):.1f}%"),
            ("Zone tấn công",
             f"{z1.get('attacking', 0):.1f}%",
             f"{z2.get('attacking', 0):.1f}%"),
        ]

        tbl_rows = []
        for i, (lbl, v1, v2) in enumerate(rows_data):
            if i == 0:
                tbl_rows.append([
                    Paragraph(f"<b>{lbl}</b>", ParagraphStyle(
                        "th", fontName="Helvetica-Bold", fontSize=9,
                        textColor=WHITE)),
                    Paragraph(f"<b>{v1}</b>", ParagraphStyle(
                        "th1", fontName="Helvetica-Bold", fontSize=9,
                        textColor=WHITE, alignment=TA_CENTER)),
                    Paragraph(f"<b>{v2}</b>", ParagraphStyle(
                        "th2", fontName="Helvetica-Bold", fontSize=9,
                        textColor=WHITE, alignment=TA_CENTER)),
                ])
            else:
                tbl_rows.append([
                    Paragraph(lbl, self.S["comp_label"]),
                    Paragraph(v1, ParagraphStyle(
                        "v1", fontName="Helvetica-Bold", fontSize=10,
                        textColor=TEAM1_C, alignment=TA_CENTER)),
                    Paragraph(v2, ParagraphStyle(
                        "v2", fontName="Helvetica-Bold", fontSize=10,
                        textColor=TEAM2_C, alignment=TA_CENTER)),
                ])

        cw = [7 * cm, 5 * cm, 5 * cm]
        tbl = Table(tbl_rows, colWidths=cw)
        style = [
            ("BACKGROUND",    (0, 0), (-1, 0),  DARK),
            ("BACKGROUND",    (1, 0), (1, 0),   TEAM1_C),
            ("BACKGROUND",    (2, 0), (2, 0),   TEAM2_C),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LIGHT]),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("GRID",          (0, 0), (-1, -1), 0.3, BORDER),
            ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ]
        tbl.setStyle(TableStyle(style))
        return [
            self._sec_bar("⚖️  BẢNG SO SÁNH CHI TIẾT"),
            Spacer(1, 0.2 * cm),
            tbl,
            Spacer(1, 0.4 * cm),
        ]

    def _s_comparison_text(self) -> list:
        """Phân tích so sánh text từ LLM."""
        comp = self.report.get("comparison", {})
        items = [
            ("possession_battle", "Kiểm soát bóng"),
            ("pressing_duel",     "Pressing"),
            ("space_usage",       "Sử dụng không gian"),
            ("physical_comparison","Thể lực"),
            ("key_difference",    "Khác biệt chiến thuật cốt lõi"),
        ]
        elems: list = [
            self._sec_bar("🔍  PHÂN TÍCH SO SÁNH"),
            Spacer(1, 0.2 * cm),
        ]
        for key, label in items:
            text = comp.get(key, "")
            if text:
                elems.append(Paragraph(
                    f"<b>{label}:</b>  {text}",
                    self.S["body"],
                ))
        elems.append(Spacer(1, 0.4 * cm))
        return elems

    def _s_key_players(self) -> list:
        """Cầu thủ nổi bật."""
        players = self.report.get("key_players", [])
        if not players:
            return []

        rows = [[
            Paragraph("<b>Đội</b>", ParagraphStyle(
                "kph", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE)),
            Paragraph("<b>Cầu thủ</b>", ParagraphStyle(
                "kph2", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE)),
            Paragraph("<b>Vai trò</b>", ParagraphStyle(
                "kph3", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE)),
            Paragraph("<b>Nhận xét</b>", ParagraphStyle(
                "kph4", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE)),
        ]]

        for p in players:
            team_name = p.get("team", "")
            color_p = TEAM1_C if self.t1_name in team_name or "1" in team_name else TEAM2_C
            rows.append([
                Paragraph(team_name, ParagraphStyle(
                    "kpt", fontName="Helvetica-Bold", fontSize=9,
                    textColor=color_p)),
                Paragraph(f"#{p.get('player_id', '')}", self.S["center_bold"]),
                Paragraph(p.get("role", ""), self.S["bullet"]),
                Paragraph(p.get("highlight", ""), self.S["bullet"]),
            ])

        cw = [3 * cm, 2.5 * cm, 3.5 * cm, 8 * cm]
        tbl = Table(rows, colWidths=cw)
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), DARK),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LIGHT]),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("GRID",          (0, 0), (-1, -1), 0.3, BORDER),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ]))
        return [
            self._sec_bar("🌟  CẦU THỦ NỔI BẬT"),
            Spacer(1, 0.2 * cm),
            tbl,
            Spacer(1, 0.4 * cm),
        ]

    def _s_conclusion(self) -> list:
        text = self.report.get("conclusion", "")
        box_data = [[Paragraph(text, self.S["body"])]]
        box = Table(box_data, colWidths=[self._W])
        box.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#fffbeb")),
            ("LINERIGHT",     (0, 0), (0, -1), 3, GOLD),
            ("LEFTPADDING",   (0, 0), (-1, -1), 14),
            ("TOPPADDING",    (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]))
        return [
            self._sec_bar("🏆  KẾT LUẬN", accent=GOLD),
            Spacer(1, 0.2 * cm),
            box,
            Spacer(1, 0.5 * cm),
        ]

    def _s_footer(self) -> list:
        date = datetime.now().strftime("%d/%m/%Y %H:%M")
        t = Table([[
            Paragraph(
                f"Báo cáo được tạo tự động bởi  AI Football Analysis System  •  {date}",
                self.S["footer"],
            )
        ]], colWidths=[self._W])
        t.setStyle(TableStyle([
            ("LINEABOVE",     (0, 0), (-1, 0), 0.5, BORDER),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        return [t]
