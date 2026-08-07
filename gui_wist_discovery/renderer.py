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

        left_panel_w = 230
        table = pygame.Rect(
            left_panel_w + 10, 60,
            SCREEN_WIDTH - left_panel_w - 320, SCREEN_HEIGHT - 80,
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

        # Buttons.
        mode_rect = self._render_mode_btn(state["paused"])
        reset_rect = self._render_reset_btn()

        # Right panel (scoreboard + discoveries).
        self._render_right_panel(state)

        pygame.display.flip()
        return mode_rect, reset_rect

    # ─── Header ─────────────────────────────────────────────────────────────────

    def _render_header(self, panel_x, state):
        """Timer and info text in the header area."""
        total_sec = state["compute_time"]
        hours = int(total_sec // 3600)
        minutes = int((total_sec % 3600) // 60)
        seconds = int(total_sec % 60)
        timer_surf = self.fonts["large"].render(f"{hours:02d}:{minutes:02d}:{seconds:02d}", True, TEXT_GOLD)
        timer_y = (60 - timer_surf.get_height()) // 2
        self.screen.blit(timer_surf, (panel_x, timer_y))

        info1 = f"Game {state['game_num']} | Shota {state['shota_num']}/5 | Speed: {state['speed']:.1f}x"
        info2 = f"Score: T1={state['team_scores'][0]:+d} T2={state['team_scores'][1]:+d} | SPACE: pause | ESC: quit"
        self.screen.blit(
            self.fonts["small"].render(info1, True, TEXT_WHITE),
            (panel_x - self.fonts["small"].size(info1)[0] - 10, 15),
        )
        self.screen.blit(
            self.fonts["small"].render(info2, True, TEXT_WHITE),
            (panel_x - self.fonts["small"].size(info2)[0] - 10, 32),
        )

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
        """Stop/Resume button in header."""
        btn_w, btn_h = 70, 24
        btn_x = SCREEN_WIDTH - 10 - btn_w - 95
        btn_y = (60 - btn_h) // 2
        btn = pygame.Rect(btn_x, btn_y, btn_w, btn_h)

        hover = btn.collidepoint(pygame.mouse.get_pos())
        label = "Resume" if paused else "Stop"
        bg = (255, 255, 255) if hover else (240, 240, 240)
        pygame.draw.rect(self.screen, bg, btn, border_radius=5)
        t = self._btn_font.render(label, True, (180, 30, 30))
        self.screen.blit(t, t.get_rect(center=btn.center))
        return btn

    def _render_reset_btn(self):
        """RESET Brain button in header."""
        btn_w, btn_h = 90, 24
        btn_x = SCREEN_WIDTH - 10 - btn_w
        btn_y = (60 - btn_h) // 2
        btn = pygame.Rect(btn_x, btn_y, btn_w, btn_h)

        hover = btn.collidepoint(pygame.mouse.get_pos())
        bg = (210, 50, 50) if hover else (180, 30, 30)
        pygame.draw.rect(self.screen, bg, btn, border_radius=5)
        t = self._btn_font_bold.render("RESET Brain", True, (255, 230, 50))
        self.screen.blit(t, t.get_rect(center=btn.center))
        return btn

    # ─── Right Panel (Scoreboard + Discoveries) ────────────────────────────────

    def _render_right_panel(self, state):
        """Scoreboard and discoveries panel on the right side."""
        px = SCREEN_WIDTH - 290
        panel_w = 280

        # Scoreboard box.
        score_h = 215
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
            f"Shotas learned: {state['episodes']:,}",
            f"Seeks: {state['seeks_achieved']}",
            f"Bids met: {state['bids_met']}/{state['bids_met'] + state['bids_failed']}",
            f"Epsilon: {state['epsilon']:.3f}  |  Stage: {state['opponent_stage']}",
        ]
        for line in stats_lines:
            self.screen.blit(self.fonts["medium"].render(line, True, TEXT_LIGHT), (px + 10, y))
            y += 18

        # Milestones box.
        disc_top = 60 + score_h + 8
        disc_h = SCREEN_HEIGHT - 80 - score_h - 8
        disc_rect = pygame.Rect(px, disc_top, panel_w, disc_h)
        pygame.draw.rect(self.screen, PANEL_DARK, disc_rect, border_radius=10)
        pygame.draw.rect(self.screen, (70, 60, 20), disc_rect, width=1, border_radius=10)

        self.screen.set_clip(pygame.Rect(px + 5, disc_top + 5, panel_w - 10, disc_h - 10))

        y = disc_rect.top + 10
        self.screen.blit(self.fonts["large"].render("Milestones", True, TEXT_GOLD), (px + 10, y))
        y += 22

        milestones = state["milestones_list"]
        if not milestones:
            self.screen.blit(self.fonts["medium"].render("None yet...", True, TEXT_DIM), (px + 15, y))
        else:
            disc_scroll = state.get("disc_scroll", 0)
            total = len(milestones)
            panel_text_w = panel_w - 30

            # Newest on top. Scroll = how many items from the top to skip.
            # When user scrolls down, they see older items.
            # New items at top don't shift the view because main.py
            # increments disc_scroll when new milestones arrive.
            for i in range(disc_scroll, total):
                idx = total - 1 - i  # Newest first.
                title_text, desc_text = milestones[idx]
                num = idx + 1

                if y + 38 > disc_rect.bottom - 15:
                    break

                is_latest = (i == 0 and disc_scroll == 0)
                title_color = (100, 255, 100) if is_latest else (255, 255, 255)

                title_str = f"{num}. {title_text}"
                y = self._wrap_text(self.fonts["large"], title_str, px + 10, y, panel_text_w, title_color, disc_rect.bottom - 20)

                desc_font = self.fonts["medium"]
                for line_idx, line_part in enumerate(desc_text.split("\n")):
                    line_color = (255, 255, 255) if line_idx == 0 else (200, 210, 220)
                    y = self._wrap_text(desc_font, line_part, px + 20, y, panel_text_w, line_color, disc_rect.bottom - 20)
                    if y > disc_rect.bottom - 20:
                        break
                y += 10

                if y > disc_rect.bottom - 15:
                    break

        self.screen.set_clip(None)

    # ─── Left Panel (Insights) ──────────────────────────────────────────────────

    def _render_insights_panel(self, panel_w, state):
        """Render left panel with strategic insights."""
        panel_rect = pygame.Rect(5, 60, panel_w - 10, SCREEN_HEIGHT - 80)
        pygame.draw.rect(self.screen, PANEL_DARK, panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, (20, 70, 50), panel_rect, width=1, border_radius=10)

        self.screen.set_clip(pygame.Rect(10, 65, panel_w - 20, SCREEN_HEIGHT - 90))

        y = panel_rect.top + 10
        self.screen.blit(self.fonts["large"].render("Strategic Insights", True, TEXT_GOLD), (15, y))
        y += 24

        insights = state.get("insights", [])
        if not insights:
            self.screen.blit(self.fonts["medium"].render("Training...", True, TEXT_DIM), (15, y))
            self.screen.set_clip(None)
            return

        total = len(insights)
        insight_scroll = state.get("insight_scroll", 0)
        text_w = panel_w - 35
        font = self.fonts["large"]  # 15px for readability.

        # Newest on top. Same stable scroll logic as milestones.
        for i in range(insight_scroll, total):
            if y + 18 > panel_rect.bottom - 15:
                break

            idx = total - 1 - i  # Newest first.
            text = insights[idx].lstrip("• ").strip()
            num = idx + 1
            is_latest = (i == 0 and insight_scroll == 0)

            # Split at " — " into title and description on same line.
            if " — " in text:
                title_part = text.split(" — ")[0]
                body_part = text.split(" — ", 1)[1]
            elif " - " in text:
                title_part = text.split(" - ")[0]
                body_part = text.split(" - ", 1)[1]
            else:
                title_part = text
                body_part = ""

            if body_part:
                full_text = f"{num}. {title_part}. {body_part}"
            else:
                full_text = f"{num}. {title_part}"

            # White text, latest entry slightly brighter.
            color = (255, 255, 255) if is_latest else (230, 240, 230)
            y = self._wrap_text(font, full_text, 15, y, text_w, color, panel_rect.bottom - 20)
            y += 8  # Gap.

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
