"""
Sudanese Wist — Pro PyGame UI

Run this file to launch the PyGame version of the game.
"""

VERSION = "2.2.0"

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
from gui_wist.constants import *
from gui_wist.card_renderer import create_card_surface
from gui_wist.game_screen import GameScreen


class WistApp:
    """Main PyGame application."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(f"{TITLE} v{VERSION}")
        self.clock = pygame.time.Clock()
        self.running = True

        self.fonts = {
            "title": pygame.font.SysFont("Segoe UI", 32, bold=True),
            "subtitle": pygame.font.SysFont("Segoe UI", 14),
            "large": pygame.font.SysFont("Segoe UI", 18, bold=True),
            "medium": pygame.font.SysFont("Segoe UI", 13),
        }

        self.state = "menu"  # Single combined menu screen.
        self._player_name = self._load_saved_name()
        self._name_cursor_visible = True
        self._name_cursor_timer = 0
        self._name_focused = False  # Whether the name input box is focused.

        self.game_screen = GameScreen(self.screen)

        # Card cache for menu display.
        self._menu_cards = [
            create_card_surface("A", "♠", CARD_LARGE_W, CARD_LARGE_H),
            create_card_surface("K", "♥", CARD_LARGE_W, CARD_LARGE_H),
            create_card_surface("Q", "♣", CARD_LARGE_W, CARD_LARGE_H),
            create_card_surface("J", "♦", CARD_LARGE_W, CARD_LARGE_H),
        ]

    def run(self):
        while self.running:
            self._handle_events()
            self._update()
            self._render()
            self.clock.tick(FPS)
        pygame.quit()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if self.state == "menu":
                    if self._name_focused:
                        if event.key == pygame.K_RETURN:
                            # Confirm name and start game.
                            if not self._player_name.strip():
                                self._player_name = "Omer"
                            self._apply_player_name()
                            self._start_playing()
                        elif event.key == pygame.K_BACKSPACE:
                            self._player_name = self._player_name[:-1]
                        elif event.key == pygame.K_ESCAPE:
                            self._name_focused = False
                        elif event.key == pygame.K_TAB:
                            self._name_focused = False
                        elif len(self._player_name) < 15 and event.unicode.isprintable() and event.unicode:
                            self._player_name += event.unicode
                    else:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
                        elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                            if not self._player_name.strip():
                                self._player_name = "Omer"
                            self._apply_player_name()
                            self._start_playing()
                    continue

                if self.state == "rules":
                    if event.key == pygame.K_ESCAPE:
                        self.state = "menu"

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.state == "menu":
                    self._handle_menu_click(event.pos)
                elif self.state == "rules":
                    # OK button click.
                    cx = SCREEN_WIDTH // 2
                    ok_rect = pygame.Rect(cx - 60, SCREEN_HEIGHT - 80, 120, 38)
                    if ok_rect.collidepoint(event.pos):
                        self.state = "menu"

            if self.state == "playing":
                self.game_screen.handle_event(event)

    def _handle_menu_click(self, pos):
        """Handle clicks on the combined menu screen."""
        cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        table = pygame.Rect(80, 80, SCREEN_WIDTH - 160, SCREEN_HEIGHT - 160)

        # Name input box — click to focus.
        box_rect = pygame.Rect(cx - 130, cy + 50, 260, 42)
        if box_rect.collidepoint(pos):
            self._name_focused = True
            return
        else:
            self._name_focused = False

        # Start Game button.
        btn_rect = pygame.Rect(cx - 120, cy + 115, 240, 55)
        if btn_rect.collidepoint(pos):
            if not self._player_name.strip():
                self._player_name = "Omer"
            self._apply_player_name()
            self._start_playing()
            return

        # Play Shotas button.
        shotas_rect = pygame.Rect(cx - 120, cy + 180, 240, 55)
        if shotas_rect.collidepoint(pos):
            if not self._player_name.strip():
                self._player_name = "Omer"
            self._apply_player_name()
            self._start_playing_shotas()
            return

        # Load AI Model button.
        model_rect = pygame.Rect(cx - 100, 175, 200, 30)
        if model_rect.collidepoint(pos):
            self._load_ai_model()
            return

        # How to Play link (text area).
        help_font = self.fonts["medium"]
        help_surf = help_font.render("How to Play", True, TEXT_LIGHT)
        help_rect = help_surf.get_rect(centerx=cx, y=cy + 250)
        if help_rect.collidepoint(pos):
            self.state = "rules"
            return

        # Reset Points button (bottom-left).
        reset_rect = pygame.Rect(table.left + 20, table.bottom - 50, 110, 30)
        if reset_rect.collidepoint(pos):
            self.game_screen._player_points = 0
            self.game_screen._player_games_played = 0
            self.game_screen._player_games_won = 0
            self.game_screen._save_player_stats()
            return

        # Exit button (bottom-right).
        exit_rect = pygame.Rect(table.right - 90, table.bottom - 50, 70, 30)
        if exit_rect.collidepoint(pos):
            self.running = False
            return

    def _start_playing(self):
        """Start the game from menu (5-shota game mode)."""
        self.state = "playing"
        # Reset draw if switching modes.
        if self.game_screen._shota_only_mode:
            self.game_screen._qabool_draw_done = False
        self.game_screen._shota_only_mode = False
        self.game_screen.start_game()

    def _start_playing_shotas(self):
        """Start endless shota mode — play shota after shota until exit."""
        self.state = "playing"
        # Reset draw if switching modes.
        if not self.game_screen._shota_only_mode:
            self.game_screen._qabool_draw_done = False
        self.game_screen._shota_only_mode = True
        self.game_screen.start_game()

    def _load_ai_model(self):
        """Open file dialog to load an AI model before starting the game."""
        from tkinter import Tk, filedialog
        try:
            root = Tk()
            root.withdraw()
            path = filedialog.askopenfilename(
                title="Load AI Model",
                filetypes=[("JSON Model", "*.json"), ("All files", "*.*")]
            )
            root.destroy()
            if path:
                self.game_screen._ai_model_path = path
        except Exception:
            pass

    def _apply_player_name(self):
        """Set the player name in the game screen and save it.
        If the name conflicts with an AI player name, rename that AI to Abubakr."""
        import gui_wist.game_screen as gs

        # Default AI names: {0: "Ibrahim", 1: "Gaafar", 3: "Musaab"}
        ai_defaults = {0: "Ibrahim", 1: "Gaafar", 3: "Musaab"}
        # Reset AI names first.
        for pid, default_name in ai_defaults.items():
            gs.DISPLAY_NAMES[pid] = default_name

        # If player chose an AI's name, rename that AI.
        name_lower = self._player_name.strip().lower()
        for pid, default_name in ai_defaults.items():
            if name_lower == default_name.lower():
                gs.DISPLAY_NAMES[pid] = "Abubakr"
                break

        gs.DISPLAY_NAMES[2] = self._player_name
        self._save_player_name()

    def _load_saved_name(self) -> str:
        """Load player name from stats file if it exists."""
        import json, os
        stats_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "player_stats.json")
        try:
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    data = json.load(f)
                return data.get("name", "Omer")
        except Exception:
            pass
        return "Omer"

    def _save_player_name(self):
        """Save player name to stats file."""
        import json, os
        stats_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "player_stats.json")
        try:
            data = {}
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    data = json.load(f)
            data["name"] = self._player_name
            with open(stats_file, 'w') as f:
                json.dump(data, f)
        except Exception:
            pass

    def _update(self):
        if self.state == "playing":
            self.game_screen.update()
            if self.game_screen.phase == "idle":
                # Check if full restart was requested (back to menu).
                if getattr(self.game_screen, '_restart_to_name', False):
                    self.game_screen._restart_to_name = False
                self.state = "menu"
        elif self.state == "menu":
            self._name_cursor_timer += 1
            if self._name_cursor_timer >= 30:
                self._name_cursor_timer = 0
                self._name_cursor_visible = not self._name_cursor_visible

    def _render(self):
        self.screen.fill(BG_DARK)

        if self.state == "menu":
            self._render_menu()
        elif self.state == "rules":
            self._render_rules()
        elif self.state == "playing":
            self.game_screen.render()

        pygame.display.flip()

    def _render_menu(self):
        """Render the combined menu — name entry + start game on one screen."""
        cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2

        # Table background.
        table = pygame.Rect(80, 80, SCREEN_WIDTH - 160, SCREEN_HEIGHT - 160)
        pygame.draw.rect(self.screen, TABLE_FELT, table, border_radius=20)
        pygame.draw.rect(self.screen, TABLE_BORDER, table, width=3, border_radius=20)

        # Title.
        title = self.fonts["title"].render("Sudanese Wist", True, TEXT_WHITE)
        self.screen.blit(title, title.get_rect(centerx=cx, y=110))

        sub = self.fonts["subtitle"].render("Telecom-Native Intelligence Research", True, TEXT_DIM)
        self.screen.blit(sub, sub.get_rect(centerx=cx, y=150))

        mx, my = pygame.mouse.get_pos()

        # Load AI Model button (right after title).
        model_rect = pygame.Rect(cx - 100, 175, 200, 30)
        hover_model = model_rect.collidepoint(mx, my)
        bg_model = (40, 100, 160) if hover_model else (30, 80, 130)
        pygame.draw.rect(self.screen, bg_model, model_rect, border_radius=6)
        if hover_model:
            pygame.draw.rect(self.screen, (80, 160, 220), model_rect, width=1, border_radius=6)
        model_text = self.fonts["medium"].render("Load AI Model", True, TEXT_LIGHT)
        self.screen.blit(model_text, model_text.get_rect(center=model_rect.center))
        if self.game_screen._ai_model_path:
            import os
            model_name = os.path.basename(self.game_screen._ai_model_path)
            loaded_surf = self.fonts["medium"].render(f"Loaded: {model_name[:25]}", True, TEXT_GREEN)
            self.screen.blit(loaded_surf, loaded_surf.get_rect(centerx=cx, y=208))

        # Fanned cards.
        for i, card_surf in enumerate(self._menu_cards):
            angle = (i - 1.5) * 10
            rotated = pygame.transform.rotate(card_surf, angle)
            x = cx + (i - 1.5) * 75 - rotated.get_width() // 2
            y = cy - 70 - rotated.get_height() // 2
            self.screen.blit(rotated, (x, y))

        # Name input section.
        if not self._name_focused:
            name_label = self.fonts["medium"].render("Player Name (click to edit):", True, TEXT_WHITE)
        else:
            name_label = self.fonts["medium"].render("Player Name:", True, TEXT_WHITE)
        self.screen.blit(name_label, name_label.get_rect(centerx=cx, y=cy + 30))

        box_rect = pygame.Rect(cx - 130, cy + 50, 260, 42)
        box_bg = (20, 40, 20) if not self._name_focused else (25, 50, 25)
        border_color = TEXT_GOLD if self._name_focused else TEXT_DIM
        pygame.draw.rect(self.screen, box_bg, box_rect, border_radius=8)
        pygame.draw.rect(self.screen, border_color, box_rect, width=2, border_radius=8)

        # Name text with cursor.
        name_font = pygame.font.SysFont("Segoe UI", 20, bold=True)
        display_name = self._player_name
        if self._name_focused and self._name_cursor_visible:
            display_name += "|"
        name_surf = name_font.render(display_name, True, TEXT_WHITE)
        self.screen.blit(name_surf, name_surf.get_rect(center=box_rect.center))

        # Reset Points button (bottom-left of table).
        reset_rect = pygame.Rect(table.left + 20, table.bottom - 50, 110, 30)
        hover_reset = reset_rect.collidepoint(mx, my)
        bg_reset = (80, 30, 30) if hover_reset else (50, 20, 20)
        pygame.draw.rect(self.screen, bg_reset, reset_rect, border_radius=6)
        reset_text = self.fonts["medium"].render("Reset Points", True, TEXT_WHITE)
        self.screen.blit(reset_text, reset_text.get_rect(center=reset_rect.center))

        # Exit button (bottom-right of table).
        exit_rect = pygame.Rect(table.right - 90, table.bottom - 50, 70, 30)
        hover_exit = exit_rect.collidepoint(mx, my)
        bg_exit = (100, 40, 40) if hover_exit else (70, 30, 30)
        pygame.draw.rect(self.screen, bg_exit, exit_rect, border_radius=6)
        exit_text = self.fonts["medium"].render("Exit", True, TEXT_WHITE)
        self.screen.blit(exit_text, exit_text.get_rect(center=exit_rect.center))

        # Start Game button.
        btn_rect = pygame.Rect(cx - 120, cy + 115, 240, 55)
        hover = btn_rect.collidepoint(mx, my)
        bg = (56, 142, 60) if hover else BUTTON_GREEN
        pygame.draw.rect(self.screen, bg, btn_rect, border_radius=10)
        if hover:
            pygame.draw.rect(self.screen, (100, 200, 100), btn_rect, width=2, border_radius=10)
        btn_text = self.fonts["large"].render("Start Game", True, TEXT_WHITE)
        self.screen.blit(btn_text, btn_text.get_rect(center=btn_rect.center))

        # Play Shotas button (same style).
        shotas_rect = pygame.Rect(cx - 120, cy + 180, 240, 55)
        hover_sh = shotas_rect.collidepoint(mx, my)
        bg_sh = (56, 142, 60) if hover_sh else BUTTON_GREEN
        pygame.draw.rect(self.screen, bg_sh, shotas_rect, border_radius=10)
        if hover_sh:
            pygame.draw.rect(self.screen, (100, 200, 100), shotas_rect, width=2, border_radius=10)
        sh_text = self.fonts["large"].render("Play Shotas", True, TEXT_WHITE)
        self.screen.blit(sh_text, sh_text.get_rect(center=shotas_rect.center))

        # How to Play — clickable link text.
        help_font = self.fonts["medium"]
        help_text_str = "How to Play"
        help_surf = help_font.render(help_text_str, True, TEXT_LIGHT)
        help_rect = help_surf.get_rect(centerx=cx, y=cy + 250)
        if help_rect.collidepoint(mx, my):
            help_surf = help_font.render(help_text_str, True, TEXT_WHITE)
            # Underline on hover.
            pygame.draw.line(self.screen, TEXT_WHITE,
                             (help_rect.left, help_rect.bottom),
                             (help_rect.right, help_rect.bottom))
        self.screen.blit(help_surf, help_rect)

        # Footer.
        inst = self.fonts["medium"].render("Press ENTER to start  |  ESC to quit", True, TEXT_DIM)
        self.screen.blit(inst, inst.get_rect(centerx=cx, y=SCREEN_HEIGHT - 100))

        ver_surf = self.fonts["medium"].render(f"v{VERSION}", True, TEXT_DIM)
        self.screen.blit(ver_surf, ver_surf.get_rect(x=SCREEN_WIDTH - 80, y=SCREEN_HEIGHT - 40))

    def _render_rules(self):
        """Render How to Play screen — centered text with OK button."""
        cx = SCREEN_WIDTH // 2
        self.screen.fill(BG_DARK)

        # Panel background.
        panel = pygame.Rect(100, 40, SCREEN_WIDTH - 200, SCREEN_HEIGHT - 100)
        pygame.draw.rect(self.screen, TABLE_FELT, panel, border_radius=16)
        pygame.draw.rect(self.screen, TABLE_BORDER, panel, width=2, border_radius=16)

        title_font = pygame.font.SysFont("Segoe UI", 22, bold=True)
        title = title_font.render("How to Play - Sudanese Wist", True, TEXT_GOLD)
        self.screen.blit(title, title.get_rect(centerx=cx, y=60))

        rules = [
            "",
            "TEAMS: You + Ibrahim (Team 1) vs Gaafar + Musaab (Team 2)",
            "GAME: 5 Shotas. First to 25 points wins. Seek = instant win.",
            "",
            "BIDDING (Al-Tasmiya):",
            "  Players bid 7-13 or pass. Bid = tricks your team commits to win.",
            "  Sahib Al-Qabool decides last: match, outbid, accept, or Dak.",
            "  Bid must be >= (cards in trump suit) + 3. Opening bid max 11.",
            "  Trump suit must have 7 or fewer cards.",
            "  If all pass: Dak (re-deal). Max 2 Daks per game.",
            "",
            "PLAYING:",
            "  Winner of the bid (Shooter) leads first with a trump card.",
            "  Follow suit if you can. If void, play anything.",
            "  Highest trump wins. If no trump played, highest of led suit wins.",
            "  Winner of each trick leads the next one.",
            "",
            "SCORING:",
            "  Bid met: playing team scores tricks won. Defenders score 0.",
            "  Bid failed: playing team LOSES bid value. Defenders score their tricks.",
            "  Seek (all 13 tricks): game ends immediately, that team wins.",
            "",
            "DAK (Re-deal):",
            "  Card Dak: no picture cards, or 8+ in one suit = mandatory re-deal.",
            "  Pass Dak: all pass, Qabool declares. Qabool rotates after.",
        ]

        font = pygame.font.SysFont("Segoe UI", 13)
        y = 95
        for line in rules:
            color = TEXT_WHITE if line and not line.startswith(" ") else TEXT_LIGHT
            if line.endswith(":"):
                color = TEXT_GOLD
            surf = font.render(line, True, color)
            self.screen.blit(surf, surf.get_rect(centerx=cx, y=y))
            y += 22

        # OK button to close.
        mx, my = pygame.mouse.get_pos()
        ok_rect = pygame.Rect(cx - 60, SCREEN_HEIGHT - 80, 120, 38)
        hover = ok_rect.collidepoint(mx, my)
        bg = (56, 142, 60) if hover else BUTTON_GREEN
        pygame.draw.rect(self.screen, bg, ok_rect, border_radius=10)
        if hover:
            pygame.draw.rect(self.screen, (100, 200, 100), ok_rect, width=2, border_radius=10)
        ok_text = self.fonts["large"].render("OK", True, TEXT_WHITE)
        self.screen.blit(ok_text, ok_text.get_rect(center=ok_rect.center))


if __name__ == "__main__":
    app = WistApp()
    app.run()
