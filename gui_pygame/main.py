"""
Sudanese Wist — Pro PyGame UI

Run this file to launch the PyGame version of the game.
"""

VERSION = "1.1.0"

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
from gui_pygame.constants import *
from gui_pygame.card_renderer import create_card_surface, create_card_back
from gui_pygame.game_screen import GameScreen


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

        self.state = "name_entry"  # Start with name entry.
        self._player_name = self._load_saved_name()
        self._name_cursor_visible = True
        self._name_cursor_timer = 0

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
                # Name entry state.
                if self.state == "name_entry":
                    if event.key == pygame.K_RETURN:
                        if not self._player_name.strip():
                            self._player_name = "Abubakr"
                        self._apply_player_name()
                        self.state = "menu"
                    elif event.key == pygame.K_BACKSPACE:
                        self._player_name = self._player_name[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        self._player_name = "Abubakr"
                        self._apply_player_name()
                        self.state = "menu"
                    elif len(self._player_name) < 15 and event.unicode.isprintable() and event.unicode:
                        self._player_name += event.unicode
                    continue

                if event.key == pygame.K_ESCAPE:
                    if self.state == "menu":
                        self.running = False
                    elif self.state == "rules":
                        self.state = "menu"
                elif event.key == pygame.K_SPACE or event.key == pygame.K_RETURN:
                    if self.state == "menu":
                        self._start_playing()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.state == "name_entry":
                    cx = SCREEN_WIDTH // 2
                    cy = SCREEN_HEIGHT // 2
                    btn_rect = pygame.Rect(cx - 80, cy + 60, 160, 45)
                    if btn_rect.collidepoint(event.pos):
                        if not self._player_name.strip():
                            self._player_name = "Abubakr"
                        self._apply_player_name()
                        self.state = "menu"
                    # Reset points button.
                    reset_rect = pygame.Rect(cx - 60, cy + 120, 120, 30)
                    if reset_rect.collidepoint(event.pos):
                        self.game_screen._player_points = 0
                        self.game_screen._player_games_played = 0
                        self.game_screen._player_games_won = 0
                        self.game_screen._save_player_stats()
                elif self.state == "menu":
                    cx = SCREEN_WIDTH // 2
                    cy = SCREEN_HEIGHT // 2
                    btn_rect = pygame.Rect(cx - 120, cy + 130, 240, 55)
                    if btn_rect.collidepoint(event.pos):
                        self._start_playing()
                    help_rect = pygame.Rect(cx - 120, cy + 200, 240, 40)
                    if help_rect.collidepoint(event.pos):
                        self.state = "rules"
                elif self.state == "rules":
                    self.state = "menu"

            if self.state == "playing":
                self.game_screen.handle_event(event)

    def _start_playing(self):
        """Start the game from menu."""
        self.state = "playing"
        self.game_screen.start_game()

    def _apply_player_name(self):
        """Set the player name in the game screen and save it."""
        import gui_pygame.game_screen as gs
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
                return data.get("name", "Abubakr")
        except Exception:
            pass
        return "Abubakr"

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
                self.state = "menu"
        elif self.state == "name_entry":
            self._name_cursor_timer += 1
            if self._name_cursor_timer >= 30:
                self._name_cursor_timer = 0
                self._name_cursor_visible = not self._name_cursor_visible

    def _render(self):
        self.screen.fill(BG_DARK)

        if self.state == "name_entry":
            self._render_name_entry()
        elif self.state == "menu":
            self._render_menu()
        elif self.state == "rules":
            self._render_rules()
        elif self.state == "playing":
            self.game_screen.render()

        pygame.display.flip()

    def _render_name_entry(self):
        """Render name entry screen."""
        cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2

        # Background.
        table = pygame.Rect(80, 100, SCREEN_WIDTH - 160, SCREEN_HEIGHT - 200)
        pygame.draw.rect(self.screen, TABLE_FELT, table, border_radius=20)
        pygame.draw.rect(self.screen, TABLE_BORDER, table, width=3, border_radius=20)

        # Title.
        title = self.fonts["title"].render("Enter Your Name", True, TEXT_WHITE)
        self.screen.blit(title, title.get_rect(centerx=cx, y=cy - 100))

        subtitle = self.fonts["medium"].render("Type your name or press ENTER to use default", True, TEXT_DIM)
        self.screen.blit(subtitle, subtitle.get_rect(centerx=cx, y=cy - 55))

        # Name input box.
        box_rect = pygame.Rect(cx - 150, cy - 20, 300, 50)
        pygame.draw.rect(self.screen, (20, 40, 20), box_rect, border_radius=8)
        pygame.draw.rect(self.screen, TEXT_GOLD, box_rect, width=2, border_radius=8)

        # Name text with cursor.
        name_font = pygame.font.SysFont("Segoe UI", 22, bold=True)
        display_name = self._player_name
        if self._name_cursor_visible:
            display_name += "|"
        name_surf = name_font.render(display_name, True, TEXT_WHITE)
        self.screen.blit(name_surf, name_surf.get_rect(center=box_rect.center))

        # Default hint.
        hint = self.fonts["medium"].render(f"Default: Abubakr", True, TEXT_DIM)
        self.screen.blit(hint, hint.get_rect(centerx=cx, y=cy + 38))

        # Continue button.
        btn_rect = pygame.Rect(cx - 80, cy + 60, 160, 45)
        mx, my = pygame.mouse.get_pos()
        hover = btn_rect.collidepoint(mx, my)
        bg = (56, 142, 60) if hover else BUTTON_GREEN
        pygame.draw.rect(self.screen, bg, btn_rect, border_radius=10)
        btn_text = self.fonts["large"].render("Continue", True, TEXT_WHITE)
        self.screen.blit(btn_text, btn_text.get_rect(center=btn_rect.center))

        # Reset Points button.
        reset_rect = pygame.Rect(cx - 60, cy + 120, 120, 30)
        hover_reset = reset_rect.collidepoint(mx, my)
        bg_reset = (80, 30, 30) if hover_reset else (50, 20, 20)
        pygame.draw.rect(self.screen, bg_reset, reset_rect, border_radius=6)
        reset_text = self.fonts["medium"].render("Reset Points", True, (200, 100, 100))
        self.screen.blit(reset_text, reset_text.get_rect(center=reset_rect.center))

    def _render_rules(self):
        """Render How to Play screen."""
        cx = SCREEN_WIDTH // 2
        self.screen.fill(BG_DARK)

        title_font = pygame.font.SysFont("Segoe UI", 24, bold=True)
        title = title_font.render("How to Play — Sudanese Wist", True, TEXT_GOLD)
        self.screen.blit(title, title.get_rect(centerx=cx, y=40))

        rules = [
            "",
            "TEAMS: You + Hima (Team 1) vs Gaafar + Musaab (Team 2)",
            "GAME: 5 Shotas (rounds). Each Shota = bidding + 13 tricks.",
            "",
            "BIDDING (Al-Tasmiya):",
            "  • Players bid 7–13 or pass. Bid = how many tricks your team will win.",
            "  • Sahib Al-Qabool (rotates each Shota) decides last — can match or accept.",
            "  • Pick a bid number, then choose your trump suit, then confirm.",
            "  • If all pass: Dak (re-deal). Max 2 Daks per game.",
            "",
            "PLAYING:",
            "  • Highest bidder (Shooter) leads first. Must play trump on first card.",
            "  • Follow suit if you can. If void, play anything (including trump).",
            "  • Highest trump wins. If no trump, highest of led suit wins.",
            "  • Trump is hidden until the first card is played.",
            "",
            "SCORING:",
            "  • If Shooter's team makes their bid: they score tricks won.",
            "  • If they fail: they LOSE the bid amount. Opponents score their tricks.",
            "  • Seek (all 13 tricks): special bonus!",
            "",
            "Click anywhere or press ESC to return to menu.",
        ]

        font = pygame.font.SysFont("Segoe UI", 14)
        y = 90
        for line in rules:
            color = TEXT_WHITE if line and not line.startswith(" ") else TEXT_LIGHT
            if line.endswith(":"):
                color = TEXT_GOLD
            surf = font.render(line, True, color)
            self.screen.blit(surf, (120, y))
            y += 24

    def _render_menu(self):
        cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2

        # Table background.
        table = pygame.Rect(80, 100, SCREEN_WIDTH - 160, SCREEN_HEIGHT - 200)
        pygame.draw.rect(self.screen, TABLE_FELT, table, border_radius=20)
        pygame.draw.rect(self.screen, TABLE_BORDER, table, width=3, border_radius=20)

        # Title.
        title = self.fonts["title"].render("Sudanese Wist", True, TEXT_WHITE)
        self.screen.blit(title, title.get_rect(centerx=cx, y=130))

        sub = self.fonts["subtitle"].render("Telecom-Native Intelligence Research", True, TEXT_WHITE)
        self.screen.blit(sub, sub.get_rect(centerx=cx, y=175))

        # Fanned cards.
        for i, card_surf in enumerate(self._menu_cards):
            angle = (i - 1.5) * 10
            rotated = pygame.transform.rotate(card_surf, angle)
            x = cx + (i - 1.5) * 80 - rotated.get_width() // 2
            y = cy - 40 - rotated.get_height() // 2
            self.screen.blit(rotated, (x, y))

        # Start button with hover.
        btn_rect = pygame.Rect(cx - 120, cy + 130, 240, 55)
        mx, my = pygame.mouse.get_pos()
        hover = btn_rect.collidepoint(mx, my)
        bg = (56, 142, 60) if hover else BUTTON_GREEN
        pygame.draw.rect(self.screen, bg, btn_rect, border_radius=10)
        if hover:
            pygame.draw.rect(self.screen, (100, 200, 100), btn_rect, width=2, border_radius=10)
        btn_text = self.fonts["large"].render("▶  Start Game", True, TEXT_WHITE)
        self.screen.blit(btn_text, btn_text.get_rect(center=btn_rect.center))

        # How to Play button.
        help_rect = pygame.Rect(cx - 120, cy + 200, 240, 40)
        hover_help = help_rect.collidepoint(mx, my)
        bg_help = (50, 50, 80) if hover_help else (40, 40, 60)
        pygame.draw.rect(self.screen, bg_help, help_rect, border_radius=8)
        if hover_help:
            pygame.draw.rect(self.screen, (100, 100, 180), help_rect, width=1, border_radius=8)
        help_text = self.fonts["medium"].render("📖  How to Play", True, TEXT_LIGHT)
        self.screen.blit(help_text, help_text.get_rect(center=help_rect.center))

        # Instructions.
        inst = self.fonts["medium"].render("Click button or press ENTER  |  ESC to quit", True, TEXT_DIM)
        self.screen.blit(inst, inst.get_rect(centerx=cx, y=SCREEN_HEIGHT - 120))

        # Version.
        ver_surf = self.fonts["medium"].render(f"v{VERSION}", True, TEXT_DIM)
        self.screen.blit(ver_surf, ver_surf.get_rect(x=SCREEN_WIDTH - 80, y=SCREEN_HEIGHT - 40))


if __name__ == "__main__":
    app = WistApp()
    app.run()
