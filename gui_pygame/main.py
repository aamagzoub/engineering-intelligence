"""
Sudanese Wist — Pro PyGame UI

Run this file to launch the PyGame version of the game.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
from gui_pygame.constants import *
from gui_pygame.card_renderer import create_card_surface, create_card_back, create_shadow


class WistGame:
    """Main PyGame application."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(TITLE)
        self.clock = pygame.time.Clock()
        self.running = True

        # Fonts.
        self.font_title = pygame.font.SysFont("Segoe UI", 28, bold=True)
        self.font_large = pygame.font.SysFont("Segoe UI", 18, bold=True)
        self.font_medium = pygame.font.SysFont("Segoe UI", 14)
        self.font_small = pygame.font.SysFont("Segoe UI", 11)
        self.font_card = pygame.font.SysFont("Consolas", 12, bold=True)

        # Pre-render card surfaces.
        self._card_cache = {}
        self._card_back = create_card_back()
        self._card_back_mini = create_card_back(CARD_MINI_W, CARD_MINI_H)
        self._shadow = create_shadow(CARD_WIDTH, CARD_HEIGHT)

        # Game state (placeholder for now).
        self.state = "menu"  # menu, playing, results

    def run(self):
        """Main game loop."""
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
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.state = "playing" if self.state == "menu" else "menu"

    def _update(self):
        """Update game state and animations."""
        pass

    def _render(self):
        """Render the current frame."""
        self.screen.fill(BG_DARK)

        if self.state == "menu":
            self._render_menu()
        elif self.state == "playing":
            self._render_table()

        pygame.display.flip()

    def _render_menu(self):
        """Render the main menu."""
        # Table background.
        table_rect = pygame.Rect(50, 80, SCREEN_WIDTH - 100, SCREEN_HEIGHT - 160)
        pygame.draw.rect(self.screen, TABLE_FELT, table_rect, border_radius=20)
        pygame.draw.rect(self.screen, TABLE_BORDER, table_rect, width=3, border_radius=20)

        # Title.
        title = self.font_title.render("Sudanese Wist", True, TEXT_WHITE)
        self.screen.blit(title, title.get_rect(centerx=SCREEN_WIDTH // 2, y=120))

        subtitle = self.font_medium.render("AI Laboratory — Engineering Intelligence", True, TEXT_DIM)
        self.screen.blit(subtitle, subtitle.get_rect(centerx=SCREEN_WIDTH // 2, y=160))

        # Demo cards fanned out.
        suits = ["♠", "♥", "♣", "♦"]
        ranks = ["A", "K", "Q", "J"]
        cx = SCREEN_WIDTH // 2
        cy = SCREEN_HEIGHT // 2

        for i, (rank, suit) in enumerate(zip(ranks, suits)):
            card = self._get_card(rank, suit)
            angle = (i - 1.5) * 12
            rotated = pygame.transform.rotate(card, angle)
            x = cx + (i - 1.5) * 60 - rotated.get_width() // 2
            y = cy - 30 - rotated.get_height() // 2
            self.screen.blit(rotated, (x, y))

        # Instructions.
        inst = self.font_large.render("Press SPACE to start", True, TEXT_GOLD)
        self.screen.blit(inst, inst.get_rect(centerx=SCREEN_WIDTH // 2, y=SCREEN_HEIGHT - 150))

        esc = self.font_small.render("ESC to quit", True, TEXT_DIM)
        self.screen.blit(esc, esc.get_rect(centerx=SCREEN_WIDTH // 2, y=SCREEN_HEIGHT - 110))

    def _render_table(self):
        """Render the game table with players and cards."""
        # Green felt table.
        table_rect = pygame.Rect(30, 60, SCREEN_WIDTH - 60, SCREEN_HEIGHT - 120)
        pygame.draw.rect(self.screen, TABLE_FELT, table_rect, border_radius=15)
        pygame.draw.rect(self.screen, TABLE_BORDER, table_rect, width=2, border_radius=15)

        # Vignette effect (darken edges).
        self._draw_vignette(table_rect)

        # Player positions.
        cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2

        # Top player (P3) — face-down cards.
        self._draw_face_down_fan(cx - 100, 80, 13)

        # Left player (P4) — face-down cards (vertical).
        self._draw_face_down_fan(60, cy - 50, 13, vertical=True)

        # Right player (P2) — face-down cards (vertical).
        self._draw_face_down_fan(SCREEN_WIDTH - 140, cy - 50, 13, vertical=True)

        # Bottom (YOU) — face-up fan.
        demo_cards = [("A", "♠"), ("K", "♠"), ("Q", "♠"), ("J", "♠"), ("10", "♠"),
                      ("A", "♥"), ("K", "♥"), ("Q", "♥"),
                      ("A", "♣"), ("K", "♣"),
                      ("A", "♦"), ("K", "♦"), ("Q", "♦")]
        self._draw_hand_fan(cx, SCREEN_HEIGHT - 100, demo_cards)

        # Centre: trick area.
        centre_rect = pygame.Rect(cx - 120, cy - 80, 240, 160)
        pygame.draw.rect(self.screen, (20, 60, 20), centre_rect, border_radius=10)

        # Info text.
        info = self.font_medium.render("Press SPACE to return to menu", True, TEXT_DIM)
        self.screen.blit(info, info.get_rect(centerx=cx, y=30))

    def _draw_face_down_fan(self, x, y, count, vertical=False):
        """Draw overlapping face-down cards."""
        for i in range(count):
            if vertical:
                pos = (x, y + i * 8)
            else:
                pos = (x + i * 15, y)
            self.screen.blit(self._card_back_mini, pos)

    def _draw_hand_fan(self, cx, cy, cards):
        """Draw a fan of face-up cards centred at (cx, cy)."""
        n = len(cards)
        total_width = (n - 1) * 55 + CARD_WIDTH
        start_x = cx - total_width // 2

        for i, (rank, suit) in enumerate(cards):
            card_surf = self._get_card(rank, suit)
            x = start_x + i * 55
            y = cy - CARD_HEIGHT // 2

            # Hover effect placeholder (cards slightly raised).
            self.screen.blit(self._shadow, (x - 1, y + 2))
            self.screen.blit(card_surf, (x, y))

    def _draw_vignette(self, rect):
        """Draw a subtle vignette (dark edges) over the table."""
        # Simple corner darkening.
        vignette = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        for i in range(30):
            alpha = max(0, 40 - i * 2)
            pygame.draw.rect(vignette, (0, 0, 0, alpha),
                             pygame.Rect(i, i, rect.width - i * 2, rect.height - i * 2),
                             width=1, border_radius=15)
        self.screen.blit(vignette, rect.topleft)

    def _get_card(self, rank: str, suit: str) -> pygame.Surface:
        """Get or create a cached card surface."""
        key = f"{rank}{suit}"
        if key not in self._card_cache:
            self._card_cache[key] = create_card_surface(rank, suit)
        return self._card_cache[key]


if __name__ == "__main__":
    game = WistGame()
    game.run()
