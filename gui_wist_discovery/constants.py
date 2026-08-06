"""Constants for Wist Discovery Watcher."""

from intelligence.core.cards.suit import Suit
from intelligence.core.cards.rank import Rank

SCREEN_WIDTH = 1380
SCREEN_HEIGHT = 800
FPS = 60
TITLE = "Sudanese Wist -- Discovery Lab"

BG_DARK = (13, 27, 13)
TABLE_FELT = (30, 80, 30)
TABLE_BORDER = (15, 50, 15)
CARD_WHITE = (255, 253, 231)
TEXT_WHITE = (245, 245, 245)
TEXT_LIGHT = (200, 230, 200)
TEXT_DIM = (90, 130, 90)
TEXT_GOLD = (255, 213, 79)
TEXT_GREEN = (102, 255, 102)
TEXT_RED = (255, 100, 100)
RED_SUIT = (198, 40, 40)
BLACK_SUIT = (26, 26, 26)
PANEL_DARK = (20, 35, 20)
BUTTON_GREEN = (67, 160, 71)
BUTTON_RED = (180, 50, 50)

CARD_WIDTH = 56
CARD_HEIGHT = 80
CARD_MINI_W = 38
CARD_MINI_H = 54
CARD_SMALL_W = 32
CARD_SMALL_H = 45
CARD_RADIUS = 6

PLAYER_NAMES = {0: "Team1-A", 1: "Team2-A", 2: "Team1-B", 3: "Team2-B"}
PLAYER_COLORS = {
    0: (100, 200, 255),
    1: (255, 180, 100),
    2: (100, 200, 255),
    3: (255, 180, 100),
}

TRICK_DELAY_MS = 1500
SHOTA_DELAY_MS = 2000


# Card ordering for sorting and display.
SUIT_ORDER = {Suit.SPADES: 0, Suit.HEARTS: 1, Suit.CLUBS: 2, Suit.DIAMONDS: 3}
RANK_ORDER = {
    Rank.TWO: 2, Rank.THREE: 3, Rank.FOUR: 4, Rank.FIVE: 5, Rank.SIX: 6,
    Rank.SEVEN: 7, Rank.EIGHT: 8, Rank.NINE: 9, Rank.TEN: 10,
    Rank.JACK: 11, Rank.QUEEN: 12, Rank.KING: 13, Rank.ACE: 14,
}
