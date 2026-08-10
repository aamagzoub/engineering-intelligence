"""
Pygame rendering — all display logic for the Discovery Watcher.
"""

import pygame

from gui_wist_discovery.constants import (
    SCREEN_WIDTH, SCREEN_HEIGHT,
    BG_DARK, TABLE_FELT, TABLE_BORDER, PANEL_DARK,
    TEXT_WHITE, TEXT_LIGHT, TEXT_DIM, TEXT_GOLD,
    CARD_WIDTH, CARD_HEIGHT, CARD_MINI_W, CARD_MINI_H,
    PLAYER_NAMES, PLAYER_COLORS, TITLE,
)
from gui_wist.card_renderer import create_card_surface, create_card_back


class Renderer:
    """Handles all pygame drawing for the Discovery Watcher."""

    def __init__(self, screen, fonts):
        self.screen = screen
        self.fonts = fonts
        self._card_cache = {}
        self._mini_card_cache = {}
        self._card_back = create_card_back(CARD_WIDTH, CARD_HEIGHT)
        # Button-specific font (created once, not per frame).
        self._btn_font = pygame.font.SysFont("Segoe UI", 11)
        self._btn_font_bold = pygame.font.SysFont("Segoe UI", 11, bold=True)
        self._cat_font = pygame.font.SysFont("Segoe UI", 13, bold=True)

    def get_card_surface(self, card):
        """Get or create a cached card surface."""
        key = f"{card.rank.symbol}{card.suit.symbol}"
        if key not in self._card_cache:
            self._card_cache[key] = create_card_surface(
                card.rank.symbol, card.suit.symbol, CARD_WIDTH, CARD_HEIGHT
            )
        return self._card_cache[key]

    def render_frame(self, state):
        """
        Render a complete frame.

        Args:
            state: RenderState namedtuple/dict with all display data.
        """
        self.screen.fill(BG_DARK)

        left_panel_w = 260
        table = pygame.Rect(
            left_panel_w + 10, 60,
            SCREEN_WIDTH - left_panel_w - 280, SCREEN_HEIGHT - 80,
        )
        pygame.draw.rect(self.screen, TABLE_FELT, table, border_radius=12)
        pygame.draw.rect(self.screen, TABLE_BORDER, table, width=2, border_radius=12)

        # Title.
        t = self.fonts["title"].render(TITLE, True, TEXT_WHITE)
        self.screen.blit(t, (left_panel_w + 15, 15))

        # Left panel (insights).
        self._render_insights_panel(left_panel_w, state)

        # Timer + info header.
        panel_x = SCREEN_WIDTH - 290
        self._render_header(panel_x, state)

        # Paused indicator.
        if state["paused"]:
            ps = self.fonts["large"].render("PAUSED", True, TEXT_GOLD)
            self.screen.blit(ps, (SCREEN_WIDTH // 2 - 40, 15))

        # Players and trick.
        self._render_players(table, state["hands"])
        self._render_trick(table, state["current_trick_cards"], state["last_winner"])

        # Buttons (both rendered together now).
        mode_rect, reset_rect = self._render_mode_btn(state["paused"])

        # Right panel (scoreboard + discoveries).
        self._render_right_panel(state)

        # "Learning Agent Information" clickable label at bottom-left.
        info_label = self.fonts["small"].render("Learning Agent Information", True, TEXT_WHITE)
        info_rect = pygame.Rect(15, SCREEN_HEIGHT - 18, info_label.get_width(), info_label.get_height())
        # Underline effect on hover.
        if info_rect.collidepoint(pygame.mouse.get_pos()):
            pygame.draw.line(self.screen, TEXT_WHITE, (15, SCREEN_HEIGHT - 4),
                             (15 + info_label.get_width(), SCREEN_HEIGHT - 4))
        self.screen.blit(info_label, (15, SCREEN_HEIGHT - 18))
        state["_info_rect"] = info_rect

        # Info overlay (if toggled on).
        if state.get("show_info"):
            self._render_info_overlay(left_panel_w)

        pygame.display.flip()
        return mode_rect, reset_rect

    # ─── Header ─────────────────────────────────────────────────────────────────

    def _render_header(self, panel_x, state):
        """Timer and shotas learned on the left, vertically centered."""
        total_sec = state["compute_time"]
        hours = int(total_sec // 3600)
        minutes = int((total_sec % 3600) // 60)
        seconds = int(total_sec % 60)
        timer_surf = self.fonts["large"].render(f"{hours:02d}:{minutes:02d}:{seconds:02d}", True, TEXT_GOLD)

        episodes = state.get("episodes", 0)
        shotas_surf = self.fonts["medium"].render(f"Shotas: {episodes:,}", True, TEXT_WHITE)

        # Both vertically centered in the 60px header, stacked.
        total_h = timer_surf.get_height() + shotas_surf.get_height() + 2
        start_y = (60 - total_h) // 2
        self.screen.blit(timer_surf, (15, start_y))
        self.screen.blit(shotas_surf, (15, start_y + timer_surf.get_height() + 2))

    # ─── Players & Cards ────────────────────────────────────────────────────────

    def _render_players(self, table, hands):
        """Render 4 player hands around the table."""
        cx, cy = table.centerx, table.centery

        for pid in range(4):
            hand = hands.get(pid, [])
            color = PLAYER_COLORS[pid]
            name_surf = self.fonts["medium"].render(PLAYER_NAMES[pid], True, color)

            if pid == 0:  # Bottom.
                hand_y = table.bottom - CARD_HEIGHT - 50
                self._render_hand_h(hand, cx, hand_y, table)
                self.screen.blit(name_surf, (cx - name_surf.get_width() // 2, hand_y + CARD_HEIGHT + 6))
            elif pid == 2:  # Top.
                self.screen.blit(name_surf, (cx - name_surf.get_width() // 2, table.top + 10))
                self._render_hand_h(hand, cx, table.top + 28, table)
            elif pid == 1:  # Left.
                self.screen.blit(name_surf, (table.left + 15, cy - 155))
                self._render_hand_v(hand, table.left + 15, cy - 135)
            elif pid == 3:  # Right.
                self.screen.blit(name_surf, (table.right - 15 - name_surf.get_width(), cy - 155))
                self._render_hand_v(hand, table.right - CARD_MINI_W - 15, cy - 135)

    def _render_hand_h(self, hand, cx, y, table):
        """Render horizontal hand with fixed 13-card spacing."""
        if not hand:
            return
        overlap = min(CARD_WIDTH - 8, (table.width - 200) // 13)
        start_x = cx - (overlap * 12 + CARD_WIDTH) // 2
        for i, card in enumerate(hand):
            self.screen.blit(self.get_card_surface(card), (start_x + i * overlap, y))

    def _render_hand_v(self, hand, x, start_y):
        """Render vertical hand with fixed 16px overlap."""
        if not hand:
            return
        for i, card in enumerate(hand):
            key = f"{card.rank.symbol}{card.suit.symbol}_mini"
            if key not in self._mini_card_cache:
                self._mini_card_cache[key] = pygame.transform.smoothscale(
                    self.get_card_surface(card), (CARD_MINI_W, CARD_MINI_H)
                )
            self.screen.blit(self._mini_card_cache[key], (x, start_y + i * 16))

    def _render_trick(self, table, trick_cards, last_winner):
        """Render current trick cards in center of table."""
        if not trick_cards:
            return
        cx, cy = table.centerx, table.centery
        offsets = {0: (0, 44), 1: (-70, 0), 2: (0, -44), 3: (70, 0)}

        for pid, card in trick_cards:
            ox, oy = offsets[pid]
            self.screen.blit(
                self.get_card_surface(card),
                (cx + ox - CARD_WIDTH // 2, cy + oy - CARD_HEIGHT // 2),
            )

        if last_winner >= 0 and len(trick_cards) == 4:
            tw = self.fonts["small"].render(f"Taker: {PLAYER_NAMES[last_winner]}", True, TEXT_GOLD)
            self.screen.blit(tw, (table.x + 120, table.y + 12))

    # ─── Buttons ────────────────────────────────────────────────────────────────

    def _render_mode_btn(self, paused):
        """Stop/Resume beneath Reset Brain, both right-aligned with scoreboard right edge."""
        px = SCREEN_WIDTH - 270
        panel_w = 260
        btn_w, btn_h = 90, 22
        gap = 4
        # Right-aligned with scoreboard right edge.
        btn_right = px + panel_w
        btn_x = btn_right - btn_w

        # Reset Brain button (top).
        reset_y = 6
        reset_btn = pygame.Rect(btn_x, reset_y, btn_w, btn_h)
        hover2 = reset_btn.collidepoint(pygame.mouse.get_pos())
        bg2 = (210, 50, 50) if hover2 else (180, 30, 30)
        pygame.draw.rect(self.screen, bg2, reset_btn, border_radius=5)
        t2 = self._btn_font_bold.render("RESET Brain", True, (255, 230, 50))
        self.screen.blit(t2, t2.get_rect(center=reset_btn.center))

        # Stop/Resume button (beneath).
        stop_y = reset_y + btn_h + gap
        btn = pygame.Rect(btn_x, stop_y, btn_w, btn_h)
        hover = btn.collidepoint(pygame.mouse.get_pos())
        label = "Resume" if paused else "Stop"
        bg = (255, 255, 255) if hover else (240, 240, 240)
        pygame.draw.rect(self.screen, bg, btn, border_radius=5)
        t = self._btn_font.render(label, True, (180, 30, 30))
        self.screen.blit(t, t.get_rect(center=btn.center))

        # Labels at bottom-right, aligned with right edge of milestones box.
        label_text = "SPACE: pause  |  ESC: quit"
        label_surf = self.fonts["small"].render(label_text, True, TEXT_WHITE)
        label_x = btn_right - label_surf.get_width()
        self.screen.blit(label_surf, (label_x, SCREEN_HEIGHT - 18))

        return btn, reset_btn

    def _render_reset_btn(self):
        """Dummy — reset is now rendered inside _render_mode_btn."""
        return pygame.Rect(0, 0, 0, 0)

    # ─── Info Overlay ──────────────────────────────────────────────────────────

    def _render_info_overlay(self, left_panel_w):
        """Render the agent info overlay — 3 columns, readable font, full black background."""
        # Full black background.
        self.screen.fill((0, 0, 0))

        # Use medium font for body (15px) and large for headers.
        title_font = self.fonts["large"]
        body_font = self.fonts["medium"]

        # 3 columns layout.
        margin = 20
        gap = 15
        col_w = (SCREEN_WIDTH - 2 * margin - 2 * gap) // 3

        col_x = [margin, margin + col_w + gap, margin + 2 * (col_w + gap)]
        start_y = 30

        # Column 1: The Environment.
        y = start_y
        self.screen.blit(title_font.render("1. The Environment", True, TEXT_GOLD), (col_x[0], y))
        y += 26
        env_lines = [
            "There are 4 players in a fixed order.",
            "Each player gets 13 cards at the start.",
            "The 4 players are split into 2 teams of 2.",
            "One player each round has a special role.",
            "Before playing cards, there is a bidding phase.",
            "After bidding, players take turns playing one card each, 4 cards per trick.",
            "There are 13 tricks per round.",
            "After 13 tricks, a score is calculated.",
            "A game has 5 rounds.",
            "The first team to reach 25 points wins.",
            "If one team wins all 13 tricks, the game ends immediately.",
            "Cards are dealt randomly each round.",
            "The special role rotates each round.",
        ]
        for line in env_lines:
            y = self._wrap_text(body_font, line, col_x[0], y, col_w, TEXT_WHITE, SCREEN_HEIGHT - 40)
            y += 4

        # Column 2: What the Agent Cannot Do.
        y = start_y
        self.screen.blit(title_font.render("2. What the Agent Cannot Do", True, TEXT_GOLD), (col_x[1], y))
        y += 26
        cannot_lines = [
            "Play a card not in its hand.",
            "Play a card from a different suit when it has cards in the led suit.",
            "Play a non-trump card on the first trick if it won the bid.",
            "Bid less than 7 or more than 13.",
            "Bid less than or equal to the current highest bid (unless special role, who can match).",
            "Bid higher than its longest valid suit count plus 3.",
            "Open with a bid higher than 11.",
            "Pass on the 3rd re-deal if it has the special role.",
            "Use a suit with 8 or more cards as trump.",
        ]
        for line in cannot_lines:
            y = self._wrap_text(body_font, line, col_x[1], y, col_w, TEXT_WHITE, SCREEN_HEIGHT - 40)
            y += 4

        # Column 3: What the Agent Knows.
        y = start_y
        self.screen.blit(title_font.render("3. What the Agent Knows", True, TEXT_GOLD), (col_x[2], y))
        y += 26
        knows_lines = [
            "It has cards, each with a number (2-14) and a suit (0-3).",
            "Each turn, some cards are playable and some are not.",
            "After 13 tricks, it gets one number (positive or negative).",
            "During bidding, it can choose a number or pass.",
            "",
            "Everything else — trump power, suit following, bidding strategy, "
            "partnership coordination — must be discovered from that single number.",
        ]
        for line in knows_lines:
            if not line:
                y += 10
                continue
            y = self._wrap_text(body_font, line, col_x[2], y, col_w, TEXT_WHITE, SCREEN_HEIGHT - 40)
            y += 4

        # "Click anywhere to close" centered at the bottom.
        close_surf = body_font.render("Click anywhere to close", True, TEXT_DIM)
        close_x = (SCREEN_WIDTH - close_surf.get_width()) // 2
        self.screen.blit(close_surf, (close_x, SCREEN_HEIGHT - 25))

    # ─── Right Panel (Scoreboard + Discoveries) ────────────────────────────────

    def _render_right_panel(self, state):
        """Scoreboard and discoveries panel on the right side."""
        px = SCREEN_WIDTH - 270
        panel_w = 260

        # Scoreboard box.
        score_h = 180
        score_rect = pygame.Rect(px, 60, panel_w, score_h)
        pygame.draw.rect(self.screen, PANEL_DARK, score_rect, border_radius=10)
        pygame.draw.rect(self.screen, (40, 60, 40), score_rect, width=1, border_radius=10)

        y = score_rect.top + 10
        self.screen.blit(self.fonts["large"].render("Scoreboard", True, TEXT_WHITE), (px + 10, y))
        y += 22

        # Table header.
        col_name_w, col_shota_w, col_total_w = 55, 30, 38
        header_x = px + 10
        self.screen.blit(self.fonts["small"].render("Team", True, TEXT_DIM), (header_x, y))
        for s in range(5):
            self.screen.blit(self.fonts["small"].render(f"S{s + 1}", True, TEXT_DIM),
                             (header_x + col_name_w + s * col_shota_w, y))
        self.screen.blit(self.fonts["small"].render("Total", True, TEXT_DIM),
                         (header_x + col_name_w + 5 * col_shota_w, y))
        y += 16
        pygame.draw.line(self.screen, (80, 80, 100), (px + 10, y), (px + panel_w - 10, y))
        y += 4

        # Team rows.
        team_names = ["Team 1", "Team 2"]
        team_colors = [(100, 200, 255), (255, 160, 100)]
        for tid in range(2):
            self.screen.blit(self.fonts["small"].render(team_names[tid], True, team_colors[tid]), (header_x, y))
            for s in range(5):
                sx = header_x + col_name_w + s * col_shota_w
                if s < len(state["shota_scores"]):
                    val = state["shota_scores"][s].get(tid, 0)
                    sc_color = (100, 255, 100) if val > 0 else (255, 100, 100) if val < 0 else TEXT_LIGHT
                    self.screen.blit(self.fonts["small"].render(f"{val:+d}", True, sc_color), (sx, y))
                else:
                    self.screen.blit(self.fonts["small"].render("—", True, TEXT_DIM), (sx + 4, y))
            total_val = state["team_scores"][tid]
            tc = (100, 255, 100) if total_val > 0 else (255, 100, 100) if total_val < 0 else TEXT_LIGHT
            self.screen.blit(self.fonts["small"].render(f"{total_val:+d}", True, tc),
                             (header_x + col_name_w + 5 * col_shota_w, y))
            y += 18

        y += 8
        stats_lines = [
            f"Stage: {state['opponent_stage']}",
            f"Games won: {state.get('team1_wins', 0)}/{state.get('team2_wins', 0)}",
            f"Bids met: {state.get('bids_met_t1', 0)}/{state.get('bids_met_t2', 0)}",
            f"Seeks: {state.get('seeks_t1', 0)}/{state.get('seeks_t2', 0)}",
        ]
        for line in stats_lines:
            self.screen.blit(self.fonts["medium"].render(line, True, TEXT_LIGHT), (px + 10, y))
            y += 18

        # Milestones box.
        disc_top = 60 + score_h + 8
        disc_h = SCREEN_HEIGHT - 80 - score_h - 8
        disc_rect = pygame.Rect(px, disc_top, panel_w, disc_h)
        pygame.draw.rect(self.screen, (0, 0, 0), disc_rect, border_radius=10)
        pygame.draw.rect(self.screen, (70, 60, 20), disc_rect, width=1, border_radius=10)

        self.screen.set_clip(pygame.Rect(px + 5, disc_top + 5, panel_w - 10, disc_h - 10))

        y = disc_rect.top + 10
        self.screen.blit(self.fonts["large"].render("Milestones", True, TEXT_GOLD), (px + 10, y))
        y += 24

        milestones = state["milestones_list"]
        if not milestones:
            self.screen.blit(self.fonts["medium"].render("None yet...", True, TEXT_DIM), (px + 15, y))
        else:
            disc_scroll = state.get("disc_scroll", 0)
            total = len(milestones)
            panel_text_w = panel_w - 30
            num_font = self.fonts["large"]    # Bold for number.
            body_font = self.fonts["medium"]  # Regular for text (same as insights).

            for i in range(disc_scroll, total):
                idx = total - 1 - i  # Newest first.
                title_text, desc_text = milestones[idx]
                num = idx + 1

                if y + 20 > disc_rect.bottom - 15:
                    break

                is_latest = (i == 0 and disc_scroll == 0)

                # Number in bold gold/white.
                num_str = f"{num}. "
                num_w = num_font.size(num_str)[0]
                num_color = TEXT_GOLD if is_latest else (255, 255, 255)
                self.screen.blit(num_font.render(num_str, True, num_color), (px + 10, y))

                # Title on first line.
                color = (255, 255, 255)
                y = self._wrap_text(num_font, title_text, px + 10 + num_w, y, panel_text_w - num_w, num_color, disc_rect.bottom - 20)
                y += 2

                # Description: render each line separately (stats + blank + milestone text).
                desc_lines = desc_text.split("\n")
                for line in desc_lines:
                    stripped = line.strip()
                    if not stripped:
                        y += 6  # Blank line = spacing before milestone text.
                        continue
                    if y + 14 > disc_rect.bottom - 15:
                        break
                    y = self._wrap_text(body_font, stripped, px + 15, y, panel_text_w - 20, color, disc_rect.bottom - 20)
                    y += 1

                y += 6

                # Horizontal separator.
                if y + 10 < disc_rect.bottom - 15:
                    pygame.draw.line(self.screen, (50, 60, 30), (px + 15, y), (px + panel_w - 15, y), 1)
                    y += 8

                if y > disc_rect.bottom - 15:
                    break

        self.screen.set_clip(None)

    # ─── Left Panel (Insights) ──────────────────────────────────────────────────

    # Category colors for badges.
    _CAT_COLORS = {
        "new": (220, 160, 0),
        "bidding": (60, 130, 200),
        "trump": (200, 80, 80),
        "timing": (180, 150, 50),
        "partnership": (80, 180, 80),
        "defense": (160, 80, 180),
        "voids": (80, 180, 180),
        "counter-intuitive": (220, 120, 0),
    }
    _CONF_LABELS = None  # No longer used — confidence is +N number now.

    def _render_insights_panel(self, panel_w, state):
        """Render left panel with structured strategic insights."""
        panel_rect = pygame.Rect(5, 60, panel_w - 10, SCREEN_HEIGHT - 80)
        pygame.draw.rect(self.screen, (0, 0, 0), panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, (20, 70, 50), panel_rect, width=1, border_radius=10)

        self.screen.set_clip(pygame.Rect(10, 65, panel_w - 20, SCREEN_HEIGHT - 90))

        y = panel_rect.top + 6

        # Title with gap after.
        self.screen.blit(self.fonts["large"].render("Strategic Insights", True, TEXT_GOLD), (15, y))
        y += 28  # Gap after title before chips.

        # Font references used throughout.
        small_font = self.fonts["small"]
        body_font = self.fonts["medium"]
        num_font = self.fonts["large"]

        # Category filter chips — after title, tight spacing.
        active_filter = state.get("insight_filter", None)
        chip_rects = {}
        chip_font = self._cat_font
        categories = list(self._CAT_COLORS.keys())
        chip_x = 15
        chip_y = y

        # Count insights per category dynamically.
        all_insights = state.get("insights", [])
        cat_counts = {}
        for cat in categories:
            if cat == "new":
                cat_counts[cat] = sum(1 for ins in all_insights
                                      if isinstance(ins, dict) and ins.get("confidence", 1) < 5)
            else:
                cat_counts[cat] = sum(1 for ins in all_insights
                                      if isinstance(ins, dict) and ins.get("category") == cat
                                      and ins.get("confidence", 1) >= 5)

        for i, cat in enumerate(categories):
            color = self._CAT_COLORS[cat]
            count = cat_counts.get(cat, 0)
            label = f"{cat.upper()} ({count})"
            is_active = (active_filter == cat)
            bg = color if is_active else (30, 30, 30)
            border = color

            chip_surf = chip_font.render(label, True, (255, 255, 255))
            cw = chip_surf.get_width() + 10
            ch = chip_surf.get_height() + 6
            chip_rect = pygame.Rect(chip_x, chip_y, cw, ch)

            if chip_x + cw > panel_w - 20:
                chip_y += ch + 3
                chip_x = 15
                chip_rect = pygame.Rect(chip_x, chip_y, cw, ch)

            pygame.draw.rect(self.screen, bg, chip_rect, border_radius=4)
            pygame.draw.rect(self.screen, border, chip_rect, width=1, border_radius=4)
            text_x = chip_rect.x + (cw - chip_surf.get_width()) // 2
            text_y = chip_rect.y + (ch - chip_surf.get_height()) // 2
            self.screen.blit(chip_surf, (text_x, text_y))
            chip_rects[cat] = chip_rect
            chip_x += cw + 3

        state["_chip_rects"] = chip_rects
        y = chip_y + ch + 8

        # Confidence filter row.
        conf_filter = state.get("confidence_filter", None)
        conf_label = small_font.render("Confident:", True, TEXT_WHITE)
        self.screen.blit(conf_label, (15, y))
        conf_x = 15 + conf_label.get_width() + 6
        conf_rects = {}
        # Show available levels.
        for level in [5, 10, 20, 50, 100]:
            is_active = (conf_filter == level)
            bg = (80, 80, 80) if is_active else (30, 30, 30)
            level_surf = small_font.render(str(level), True, (255, 255, 255))
            lw = level_surf.get_width() + 6
            lh = level_surf.get_height() + 4
            level_rect = pygame.Rect(conf_x, y, lw, lh)
            pygame.draw.rect(self.screen, bg, level_rect, border_radius=3)
            pygame.draw.rect(self.screen, (100, 100, 100), level_rect, width=1, border_radius=3)
            self.screen.blit(level_surf, (conf_x + 3, y + 2))
            conf_rects[level] = level_rect
            conf_x += lw + 3
        state["_conf_rects"] = conf_rects
        y += lh + 10

        insights = state.get("insights", [])
        if not insights:
            self.screen.blit(self.fonts["medium"].render("Training...", True, TEXT_DIM), (15, y))
            self.screen.set_clip(None)
            return

        # Filter insights by category if a filter is active.
        if active_filter:
            if active_filter == "new":
                insights = [ins for ins in insights
                            if (isinstance(ins, dict) and ins.get("confidence", 1) < 5)]
            else:
                insights = [ins for ins in insights
                            if (isinstance(ins, dict) and ins.get("category") == active_filter
                                and ins.get("confidence", 1) >= 5)]

        # Filter by confidence level.
        if conf_filter:
            insights = [ins for ins in insights
                        if (isinstance(ins, dict) and ins.get("confidence", 1) >= conf_filter)]

        total = len(insights)
        insight_scroll = state.get("insight_scroll", 0)
        text_w = panel_w - 35

        for i in range(insight_scroll, total):
            if y + 20 > panel_rect.bottom - 15:
                break

            idx = total - 1 - i
            ins = insights[idx]

            # Handle both old string format and new dict format.
            if isinstance(ins, str):
                text = ins.lstrip("• ").strip()
                category = ""
                confidence = 1
                condition = ""
                why = ""
                is_new = False
            else:
                text = ins.get("text", "")
                category = ins.get("category", "")
                confidence = ins.get("confidence", 1)
                condition = ins.get("condition", "")
                why = ins.get("why", "")
                is_new = ins.get("new", False)

            num = idx + 1
            is_latest = (i == 0 and insight_scroll == 0)

            # --- Line 1: Number + version + category badge + NEW badge + confidence ---
            x_cursor = 15

            # Number with version.
            version = ins.get("version", 0) if isinstance(ins, dict) else 0
            version_str = f" (+{version})" if version > 0 else ""
            num_str = f"{num}{version_str}. "
            num_color = (255, 255, 255)
            self.screen.blit(num_font.render(num_str, True, num_color), (x_cursor, y))
            x_cursor += num_font.size(num_str)[0]

            # Badge logic:
            # Confidence 1-4: NEW badge only (no category, no +N)
            # Confidence 5+: Category badge + +N (no NEW)
            conf_val = ins.get("confidence", 1) if isinstance(ins, dict) else 1
            if conf_val < 5:
                # NEW badge only.
                new_surf = small_font.render("NEW", True, (0, 0, 0))
                new_w = new_surf.get_width() + 8
                new_h = new_surf.get_height() + 4
                new_rect = pygame.Rect(x_cursor, y + (18 - new_h) // 2, new_w, new_h)
                pygame.draw.rect(self.screen, (220, 160, 0), new_rect, border_radius=3)
                self.screen.blit(new_surf, (x_cursor + 4, new_rect.y + 2))
                x_cursor += new_w + 4
            else:
                # Category badge.
                if category:
                    cat_color = self._CAT_COLORS.get(category, (120, 120, 120))
                    cat_text = category.upper()
                    cat_surf = small_font.render(cat_text, True, (255, 255, 255))
                    cat_w = cat_surf.get_width() + 8
                    cat_h = cat_surf.get_height() + 4
                    badge_rect = pygame.Rect(x_cursor, y + (18 - cat_h) // 2, cat_w, cat_h)
                    pygame.draw.rect(self.screen, cat_color, badge_rect, border_radius=3)
                    self.screen.blit(cat_surf, (x_cursor + 4, badge_rect.y + 2))
                    x_cursor += cat_w + 4
                # +N confidence.
                conf_text = f"+{conf_val - 1}"
                self.screen.blit(small_font.render(conf_text, True, (255, 255, 255)), (x_cursor, y + 3))
                x_cursor += small_font.size(conf_text)[0] + 4

            y += 18

            # --- Line 2+: Main text + why combined + discovery shota ---
            full_text = text
            if why:
                full_text = f"{text}. {why}"
            episode = ins.get("episode", 0) if isinstance(ins, dict) else 0
            if episode > 0:
                full_text += f" (at {episode:,} shotas)"
            y = self._wrap_text(body_font, full_text, 25, y, text_w - 10, (255, 255, 255), panel_rect.bottom - 20)
            y += 2

            y += 4

            # Separator.
            if y + 10 < panel_rect.bottom - 15:
                pygame.draw.line(self.screen, (40, 70, 40), (20, y), (panel_w - 30, y), 1)
                y += 8

            if y > panel_rect.bottom - 15:
                break

        self.screen.set_clip(None)

    # ─── Utilities ──────────────────────────────────────────────────────────────

    def _wrap_text(self, font, text, x, y, max_w, color, max_y):
        """Word-wrap text, rendering line by line. Returns new y position."""
        words = text.split()
        line = ""
        for w in words:
            test = line + " " + w if line else w
            if font.size(test)[0] < max_w:
                line = test
            else:
                if y > max_y:
                    return y
                self.screen.blit(font.render(line, True, color), (x, y))
                y += 16
                line = w
        if line and y <= max_y:
            self.screen.blit(font.render(line, True, color), (x, y))
            y += 16
        return y
