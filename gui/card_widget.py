"""
Card widget — draws realistic playing cards using tkinter Canvas.

Each card is a white rounded rectangle with:
- Rank in top-left and bottom-right corners
- Large suit symbol in the center
- Red for hearts/diamonds, black for spades/clubs
- Optional highlight (green border for playable, gold for selected)
"""

import tkinter as tk


# Card dimensions.
CARD_WIDTH = 42
CARD_HEIGHT = 60
CARD_MINI_WIDTH = 34
CARD_MINI_HEIGHT = 50
CARD_LARGE_WIDTH = 56
CARD_LARGE_HEIGHT = 78

# Colors.
CARD_BG = "#ffffff"
CARD_BORDER = "#888888"
CARD_RED = "#c62828"
CARD_BLACK = "#1a1a1a"
CARD_HIGHLIGHT = "#4caf50"
CARD_SELECTED = "#ffd54f"
CARD_DISABLED = "#cccccc"
CARD_BACK = "#1565c0"
CARD_BACK_PATTERN = "#1976d2"


def get_suit_color(suit_symbol: str) -> str:
    """Return red or black based on suit."""
    if suit_symbol in ("♥", "♦"):
        return CARD_RED
    return CARD_BLACK


def draw_card(canvas: tk.Canvas, x: int, y: int, rank: str, suit: str,
              width: int = CARD_WIDTH, height: int = CARD_HEIGHT,
              highlight: str | None = None, faded: bool = False,
              tag: str = "") -> None:
    """
    Draw a single playing card on a canvas.

    Args:
        canvas: The tkinter Canvas to draw on.
        x, y: Top-left corner position.
        rank: Card rank text ("A", "K", "Q", "J", "10", "9", etc.)
        suit: Suit symbol ("♠", "♥", "♦", "♣")
        width, height: Card dimensions.
        highlight: Border color for highlighting (None = default border).
        faded: If True, draw with muted colors (for illegal cards).
        tag: Canvas tag for this card (for click binding).
    """
    # Card background.
    border_color = highlight or CARD_BORDER
    border_width = 2 if highlight else 1

    # Rounded rectangle (approximate with rectangle + oval corners).
    r = 4  # corner radius
    canvas.create_rectangle(
        x + r, y, x + width - r, y + height,
        fill=CARD_BG if not faded else CARD_DISABLED,
        outline=border_color, width=border_width, tags=tag)
    canvas.create_rectangle(
        x, y + r, x + width, y + height - r,
        fill=CARD_BG if not faded else CARD_DISABLED,
        outline=border_color, width=border_width, tags=tag)
    # Fill corners.
    canvas.create_oval(x, y, x + r * 2, y + r * 2,
                       fill=CARD_BG if not faded else CARD_DISABLED,
                       outline=border_color, width=border_width, tags=tag)
    canvas.create_oval(x + width - r * 2, y, x + width, y + r * 2,
                       fill=CARD_BG if not faded else CARD_DISABLED,
                       outline=border_color, width=border_width, tags=tag)
    canvas.create_oval(x, y + height - r * 2, x + r * 2, y + height,
                       fill=CARD_BG if not faded else CARD_DISABLED,
                       outline=border_color, width=border_width, tags=tag)
    canvas.create_oval(x + width - r * 2, y + height - r * 2, x + width, y + height,
                       fill=CARD_BG if not faded else CARD_DISABLED,
                       outline=border_color, width=border_width, tags=tag)
    # Inner fill (cover the seams).
    canvas.create_rectangle(
        x + 2, y + 2, x + width - 2, y + height - 2,
        fill=CARD_BG if not faded else CARD_DISABLED, outline="", tags=tag)

    # Text color.
    color = get_suit_color(suit)
    if faded:
        color = "#999999"

    # Rank in top-left.
    font_size = 9 if width >= CARD_WIDTH else 7
    canvas.create_text(
        x + 6, y + 8, text=rank, anchor="nw",
        font=("Consolas", font_size, "bold"), fill=color, tags=tag)

    # Large suit symbol in center.
    suit_size = 16 if width >= CARD_LARGE_WIDTH else (13 if width >= CARD_WIDTH else 11)
    canvas.create_text(
        x + width // 2, y + height // 2 + 4, text=suit, anchor="center",
        font=("Segoe UI", suit_size), fill=color, tags=tag)

    # Rank in bottom-right (upside down effect - just smaller).
    canvas.create_text(
        x + width - 6, y + height - 8, text=rank, anchor="se",
        font=("Consolas", font_size - 1), fill=color, tags=tag)


def draw_card_back(canvas: tk.Canvas, x: int, y: int,
                   width: int = CARD_WIDTH, height: int = CARD_HEIGHT,
                   tag: str = "") -> None:
    """Draw a face-down card (card back)."""
    canvas.create_rectangle(
        x, y, x + width, y + height,
        fill=CARD_BACK, outline="#0d47a1", width=1, tags=tag)
    # Pattern (diamond).
    cx, cy = x + width // 2, y + height // 2
    canvas.create_rectangle(
        cx - 8, cy - 12, cx + 8, cy + 12,
        fill=CARD_BACK_PATTERN, outline="#1565c0", tags=tag)


def parse_card_text(card_text: str) -> tuple[str, str]:
    """
    Parse a card text like "A♠" or "10♥" into (rank, suit).
    Returns ("A", "♠") or ("10", "♥").
    """
    suits = "♠♥♦♣"
    for i, ch in enumerate(card_text):
        if ch in suits:
            return card_text[:i], card_text[i:]
    return card_text, "?"


class CardCanvas(tk.Canvas):
    """
    A Canvas that displays a row of cards.
    Used in player hand areas and trick centres.
    """

    def __init__(self, parent, card_size="normal", bg="#1a3a1a", **kwargs):
        self._card_w = {"small": CARD_MINI_WIDTH, "normal": CARD_WIDTH, "large": CARD_LARGE_WIDTH}[card_size]
        self._card_h = {"small": CARD_MINI_HEIGHT, "normal": CARD_HEIGHT, "large": CARD_LARGE_HEIGHT}[card_size]

        height = self._card_h + 8
        super().__init__(parent, bg=bg, height=height, highlightthickness=0, **kwargs)

    def show_cards(self, card_texts: list[str], highlights: dict[int, str] | None = None,
                   faded_indices: set[int] | None = None, on_click=None) -> None:
        """
        Display cards in a row.

        Args:
            card_texts: List of card text strings like ["A♠", "K♥", "10♦"]
            highlights: {index: color} for highlighted cards.
            faded_indices: Set of indices to fade (illegal cards).
            on_click: Callback(index) when a card is clicked.
        """
        self.delete("all")
        highlights = highlights or {}
        faded_indices = faded_indices or set()

        spacing = min(self._card_w + 4, max(20, (self.winfo_width() - 20) // max(len(card_texts), 1)))
        start_x = 4

        for i, ct in enumerate(card_texts):
            rank, suit = parse_card_text(ct)
            x = start_x + i * spacing
            y = 4

            hl = highlights.get(i)
            faded = i in faded_indices
            tag = f"card_{i}"

            draw_card(self, x, y, rank, suit,
                      width=self._card_w, height=self._card_h,
                      highlight=hl, faded=faded, tag=tag)

            if on_click:
                self.tag_bind(tag, "<Button-1>", lambda e, idx=i: on_click(idx))

    def show_empty(self, text: str = "No cards") -> None:
        """Show placeholder text."""
        self.delete("all")
        w = self.winfo_width() or 200
        h = self.winfo_height() or 50
        self.create_text(w // 2, h // 2, text=text, fill="#5a8a5a", font=("Segoe UI", 9))

    def show_card_backs(self, count: int) -> None:
        """Show face-down cards (won tricks pile)."""
        self.delete("all")
        spacing = min(16, max(8, 100 // max(count, 1)))
        for i in range(count):
            draw_card_back(self, 4 + i * spacing, 4,
                           width=self._card_w, height=self._card_h)
