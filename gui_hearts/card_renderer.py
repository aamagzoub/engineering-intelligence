"""Card rendering for Hearts PyGame GUI."""

import pygame
from gui_hearts.constants import CARD_WIDTH, CARD_HEIGHT, CARD_RADIUS, CARD_WHITE, RED_SUIT, BLACK_SUIT


def create_card_surface(rank: str, suit: str, width: int = CARD_WIDTH,
                        height: int = CARD_HEIGHT) -> pygame.Surface:
    """Create a card surface with rank and suit."""
    surf = pygame.Surface((width, height), pygame.SRCALPHA)

    # Card body.
    rect = pygame.Rect(0, 0, width, height)
    pygame.draw.rect(surf, CARD_WHITE, rect, border_radius=CARD_RADIUS)
    pygame.draw.rect(surf, (180, 180, 180), rect, width=1, border_radius=CARD_RADIUS)

    # Suit color.
    color = RED_SUIT if suit in ("♥", "♦") else BLACK_SUIT

    # Rank in top-left.
    font_size = max(11, width // 5)
    font = pygame.font.SysFont("Consolas", font_size, bold=True)
    rank_surf = font.render(rank, True, color)
    surf.blit(rank_surf, (4, 2))

    # Suit below rank.
    suit_font = pygame.font.SysFont("Segoe UI", font_size - 2)
    suit_small = suit_font.render(suit, True, color)
    surf.blit(suit_small, (4, 2 + font_size))

    # Large suit in center.
    big_font = pygame.font.SysFont("Segoe UI", max(18, width // 2))
    big_suit = big_font.render(suit, True, color)
    big_rect = big_suit.get_rect(center=(width // 2, height // 2 + 4))
    surf.blit(big_suit, big_rect)

    return surf


def create_card_back(width: int = CARD_WIDTH, height: int = CARD_HEIGHT) -> pygame.Surface:
    """Card back — red/maroon theme for Hearts."""
    surf = pygame.Surface((width, height), pygame.SRCALPHA)
    rect = pygame.Rect(0, 0, width, height)
    pygame.draw.rect(surf, (120, 20, 40), rect, border_radius=CARD_RADIUS)

    # Inner border.
    inner = pygame.Rect(3, 3, width - 6, height - 6)
    pygame.draw.rect(surf, (140, 30, 50), inner, border_radius=4)
    pygame.draw.rect(surf, (180, 50, 70), inner, width=1, border_radius=4)

    # Heart in center.
    cx, cy = width // 2, height // 2
    heart_font = pygame.font.SysFont("Segoe UI", max(14, width // 3))
    heart_surf = heart_font.render("♥", True, (200, 80, 80, 120))
    heart_rect = heart_surf.get_rect(center=(cx, cy))
    surf.blit(heart_surf, heart_rect)

    pygame.draw.rect(surf, (100, 15, 30), rect, width=2, border_radius=CARD_RADIUS)
    return surf
