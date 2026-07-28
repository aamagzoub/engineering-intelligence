"""
Sudanese Wist — Pro PyGame UI

Run this file to launch the PyGame version of the game.
"""

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
        pygame.display.set_caption(TITLE)
        self.clock = pygame.time.Clock()
        self.running = True

        self.fonts = {
            "title": pygame.font.SysFont("Segoe UI", 32, bold=True),
            "subtitle": pygame.font.SysFont("Segoe UI", 14),
            "large": pygame.font.SysFont("Segoe UI", 18, bold=True),
            "medium": pygame.font.SysFont("Segoe UI", 13),
        }

        self.state = "menu"
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
                if event.key == pygame.K_ESCAPE:
                    if self.state == "menu":
                        self.running = False
                    else:
                        self.state = "menu"
                elif event.key == pygame.K_SPACE or event.key == pygame.K_RETURN:
                    if self.state == "menu":
                        self._start_playing()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.state == "menu":
                    # Check Start Game button click.
                    cx = SCREEN_WIDTH // 2
                    cy = SCREEN_HEIGHT // 2
                    btn_rect = pygame.Rect(cx - 120, cy + 130, 240, 55)
                    if btn_rect.collidepoint(event.pos):
                        self._start_playing()

            if self.state == "playing":
                self.game_screen.handle_event(event)

    def _start_playing(self):
        """Start the game from menu."""
        self.state = "playing"
        self.game_screen.start_game()

    def _update(self):
        if self.state == "playing":
            self.game_screen.update()

    def _render(self):
        self.screen.fill(BG_DARK)

        if self.state == "menu":
            self._render_menu()
        elif self.state == "playing":
            self.game_screen.render()

        pygame.display.flip()

    def _render_menu(self):
        cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2

        # Table background.
        table = pygame.Rect(80, 100, SCREEN_WIDTH - 160, SCREEN_HEIGHT - 200)
        pygame.draw.rect(self.screen, TABLE_FELT, table, border_radius=20)
        pygame.draw.rect(self.screen, TABLE_BORDER, table, width=3, border_radius=20)

        # Title.
        title = self.fonts["title"].render("Sudanese Wist", True, TEXT_WHITE)
        self.screen.blit(title, title.get_rect(centerx=cx, y=130))

        sub = self.fonts["subtitle"].render("AI Laboratory — Engineering Intelligence Research", True, TEXT_DIM)
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

        # Instructions.
        inst = self.fonts["medium"].render("Click button or press ENTER  |  ESC to quit", True, TEXT_DIM)
        self.screen.blit(inst, inst.get_rect(centerx=cx, y=SCREEN_HEIGHT - 120))


if __name__ == "__main__":
    app = WistApp()
    app.run()
