"""
Card rendering for PyGame — draws beautiful playing cards.
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
    """Create a card back surface."""
    surf = pygame.Surface((width, height), pygame.SRCALPHA)

    # Blue card back.
    rect = pygame.Rect(0, 0, width, height)
    pygame.draw.rect(surf, (21, 101, 192), rect, border_radius=CARD_RADIUS)
    pygame.draw.rect(surf, (13, 71, 161), rect, width=1, border_radius=CARD_RADIUS)

    # Diamond pattern.
    inner = pygame.Rect(6, 6, width - 12, height - 12)
    pygame.draw.rect(surf, (25, 118, 210), inner, border_radius=4)

    # Centre diamond.
    cx, cy = width // 2, height // 2
    points = [(cx, cy - 15), (cx + 10, cy), (cx, cy + 15), (cx - 10, cy)]
    pygame.draw.polygon(surf, (255, 255, 255, 100), points)

    return surf


def create_shadow(width: int, height: int, offset: int = 3) -> pygame.Surface:
    """Create a card shadow surface."""
    surf = pygame.Surface((width + offset * 2, height + offset * 2), pygame.SRCALPHA)
    shadow_rect = pygame.Rect(offset, offset, width, height)
    pygame.draw.rect(surf, (0, 0, 0, 50), shadow_rect, border_radius=CARD_RADIUS)
    return surf
