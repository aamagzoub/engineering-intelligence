from intelligence.core.cards.deck import Deck
from environments.wist.player import Player


class Round:
    """
    Controls one round of Wist.
    """

    def __init__(self, players: list[Player]) -> None:
        if len(players) != 4:
            raise ValueError("A Wist round requires exactly 4 players.")

        self.players = players
        self.deck = Deck()

    def deal(self) -> None:
        """
        Shuffle and deal 13 cards to each player.
        """

        self.deck.shuffle()

        # Clear any previous hands
        for player in self.players:
            player.hand.clear()

        # Counter-clockwise dealing (one card at a time)
        for _ in range(13):
            for player in self.players:
                player.receive_cards(
                    [self.deck.deal(1)[0]]
                )