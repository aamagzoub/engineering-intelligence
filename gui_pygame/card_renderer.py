"""
Card rendering for PyGame — draws playing cards programmatically.
"""

import pygame
from gui_pygame.constants import *


def create_card_surface(rank: str, suit: str, width: int = CARD_WIDTH,
                        height: int = CARD_HEIGHT) -> pygame.Surface:
    """Create a card surface with proper styling."""
    surf = pygame.Surface((width, height), pygame.SRCALPHA)

    # Card body (rounded rectangle).
    rect = pygame.Rect(0, 0, width, height)
    pygame.draw.rect(surf, CARD_WHITE, rect, border_radius=CARD_RADIUS)
    pygame.draw.rect(surf, (180, 180, 180), rect, width=1, border_radius=CARD_RADIUS)

    # Suit color.
    color = RED_SUIT if suit in ("♥", "♦") else BLACK_SUIT

    # Rank in top-left.
    font_size = max(12, width // 5)
    font = pygame.font.SysFont("Consolas", font_size, bold=True)
    rank_surf = font.render(rank, True, color)
    surf.blit(rank_surf, (5, 3))

    # Suit symbol in top-left (below rank).
    suit_font = pygame.font.SysFont("Segoe UI", font_size - 2)
    suit_small = suit_font.render(suit, True, color)
    surf.blit(suit_small, (5, 3 + font_size))

    # Large suit in centre.
    big_font = pygame.font.SysFont("Segoe UI", max(20, width // 2))
    big_suit = big_font.render(suit, True, color)
    big_rect = big_suit.get_rect(center=(width // 2, height // 2 + 5))
    surf.blit(big_suit, big_rect)

    # Rank in bottom-right (rotated).
    rank_br = font.render(rank, True, color)
    rank_br = pygame.transform.rotate(rank_br, 180)
    surf.blit(rank_br, (width - rank_br.get_width() - 5, height - rank_br.get_height() - 3))

    return surf


def create_card_back(width: int = CARD_WIDTH, height: int = CARD_HEIGHT) -> pygame.Surface:
    """Create a card back with a repeating geometric lattice pattern."""
    surf = pygame.Surface((width, height), pygame.SRCALPHA)

    # Base — deep blue rounded rect.
    rect = pygame.Rect(0, 0, width, height)
    pygame.draw.rect(surf, (18, 60, 140), rect, border_radius=CARD_RADIUS)

    # Inner border frame.
    inner = pygame.Rect(4, 4, width - 8, height - 8)
    pygame.draw.rect(surf, (22, 80, 170), inner, border_radius=5)
    pygame.draw.rect(surf, (40, 110, 200), inner, width=1, border_radius=5)

    # Repeating diamond lattice pattern inside the inner frame.
    pattern_rect = pygame.Rect(7, 7, width - 14, height - 14)
    pattern_surf = pygame.Surface((pattern_rect.width, pattern_rect.height), pygame.SRCALPHA)

    # Diamond grid — small repeating diamonds.
    diamond_size = 8
    line_color = (60, 140, 230, 70)
    for row in range(0, pattern_rect.height + diamond_size, diamond_size):
        for col in range(0, pattern_rect.width + diamond_size, diamond_size * 2):
            offset = diamond_size if (row // diamond_size) % 2 else 0
            cx = col + offset
            cy = row
            # Small diamond.
            half = diamond_size // 2
            pts = [(cx, cy - half), (cx + half, cy), (cx, cy + half), (cx - half, cy)]
            pygame.draw.polygon(pattern_surf, line_color, pts, width=1)

    # Overlay a subtle cross-hatch.
    hatch_color = (80, 160, 240, 40)
    for i in range(-pattern_rect.height, pattern_rect.width, 12):
        pygame.draw.line(pattern_surf, hatch_color, (i, 0), (i + pattern_rect.height, pattern_rect.height))
        pygame.draw.line(pattern_surf, hatch_color, (i + pattern_rect.height, 0), (i, pattern_rect.height))

    surf.blit(pattern_surf, pattern_rect.topleft)

    # Centre ornament — larger diamond with inner detail.
    cx, cy = width // 2, height // 2
    # Outer diamond.
    size = min(width, height) // 4
    outer_pts = [(cx, cy - size), (cx + size, cy), (cx, cy + size), (cx - size, cy)]
    pygame.draw.polygon(surf, (255, 255, 255, 50), outer_pts)
    pygame.draw.polygon(surf, (180, 220, 255, 100), outer_pts, width=1)
    # Inner diamond.
    s2 = size // 2
    inner_pts = [(cx, cy - s2), (cx + s2, cy), (cx, cy + s2), (cx - s2, cy)]
    pygame.draw.polygon(surf, (255, 255, 255, 30), inner_pts)
    pygame.draw.polygon(surf, (180, 220, 255, 80), inner_pts, width=1)

    # Card edge border.
    pygame.draw.rect(surf, (13, 50, 120), rect, width=2, border_radius=CARD_RADIUS)

    return surf
