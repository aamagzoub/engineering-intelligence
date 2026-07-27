"""
Controller helper methods — display formatting, card sorting, etc.

These are mixed into the SimulationController via inheritance.
"""

from environments.wist.rules import rank_value


class ControllerHelpersMixin:
    """Display and formatting helper methods for the controller."""

    def show_player_hands(self) -> None:
        if self.players is None:
            return

        for index, player in enumerate(self.players):
            cards = self.get_cards_from_player(player)
            sorted_cards = self.sort_cards(cards)
            card_labels = [self.format_card_short(card) for card in sorted_cards]
            self.app.set_player_hand(index, card_labels)

    def get_cards_from_player(self, player):
        if hasattr(player, "hand"):
            hand = player.hand
            if hasattr(hand, "cards"):
                return hand.cards
            if isinstance(hand, list):
                return hand
        if hasattr(player, "cards"):
            return player.cards
        return []

    def log_trick_cards(self, trick) -> None:
        played_cards = self.extract_played_cards_from_trick(trick)
        if not played_cards:
            return

        visual_played_cards = []
        for player_id, card in played_cards:
            short_card_text = self.format_card_short(card)
            visual_played_cards.append((player_id, short_card_text))

        self.app.set_played_cards(visual_played_cards)

    def extract_played_cards_from_trick(self, trick):
        possible_attributes = ["played_cards", "cards_played", "plays", "cards", "actions"]

        for attribute_name in possible_attributes:
            if not hasattr(trick, attribute_name):
                continue
            value = getattr(trick, attribute_name)

            if isinstance(value, dict):
                return [(player_id, card) for player_id, card in value.items()]

            if isinstance(value, list):
                extracted = []
                for item in value:
                    if isinstance(item, tuple) and len(item) >= 2:
                        extracted.append((item[0], item[1]))
                    elif isinstance(item, list) and len(item) >= 2:
                        extracted.append((item[0], item[1]))
                    elif hasattr(item, "player_id") and hasattr(item, "card"):
                        extracted.append((item.player_id, item.card))
                if extracted:
                    return extracted
        return []

    def sort_cards(self, cards):
        suit_order = {"SPADES": 0, "HEARTS": 1, "CLUBS": 2, "DIAMONDS": 3}
        rank_order = {
            "ACE": 0, "KING": 1, "QUEEN": 2, "JACK": 3, "TEN": 4,
            "NINE": 5, "EIGHT": 6, "SEVEN": 7, "SIX": 8, "FIVE": 9,
            "FOUR": 10, "THREE": 11, "TWO": 12,
        }

        def card_key(card):
            suit = getattr(card, "suit", None)
            rank = getattr(card, "rank", None)
            suit_name = getattr(suit, "name", str(suit)).upper()
            rank_name = getattr(rank, "name", str(rank)).upper()
            return (suit_order.get(suit_name, 99), rank_order.get(rank_name, 99))

        return sorted(cards, key=card_key)

    def format_card(self, card) -> str:
        rank = getattr(card, "rank", None)
        suit = getattr(card, "suit", None)
        rank_text = getattr(rank, "name", str(rank))
        suit_text = getattr(suit, "name", str(suit))
        return f"{rank_text} of {suit_text}"

    def _format_trump(self, trump_suit) -> str:
        if trump_suit is None:
            return "-"
        suit_symbols = {"SPADES": "♠", "HEARTS": "♥", "CLUBS": "♣", "DIAMONDS": "♦"}
        name = getattr(trump_suit, "name", str(trump_suit))
        symbol = suit_symbols.get(name.upper(), "")
        short_names = {"SPADES": "Spd", "HEARTS": "Hrt", "CLUBS": "Clb", "DIAMONDS": "Dmd"}
        short = short_names.get(name.upper(), name)
        return f"{symbol} {short}"

    def format_card_short(self, card) -> str:
        rank = getattr(card, "rank", None)
        suit = getattr(card, "suit", None)
        rank_name = getattr(rank, "name", str(rank)).upper()
        suit_name = getattr(suit, "name", str(suit)).upper()

        rank_map = {
            "ACE": "A", "KING": "K", "QUEEN": "Q", "JACK": "J", "TEN": "10",
            "NINE": "9", "EIGHT": "8", "SEVEN": "7", "SIX": "6", "FIVE": "5",
            "FOUR": "4", "THREE": "3", "TWO": "2",
        }
        suit_map = {"SPADES": "♠", "HEARTS": "♥", "DIAMONDS": "♦", "CLUBS": "♣"}

        rank_text = rank_map.get(rank_name, rank_name[:1])
        suit_text = suit_map.get(suit_name, "?")
        return f"{rank_text}{suit_text}"

    def get_team_index(self, player_index: int) -> int:
        if player_index in [0, 2]:
            return 0
        return 1

    def set_shota_info_safe(self, trump: str, qabool: str = "-",
                            bid: str = "-", first_shooter: str = "-") -> None:
        try:
            self.app.set_shota_info(trump=trump, qabool=qabool, bid=bid,
                                    first_shooter=first_shooter)
        except TypeError:
            self.app.set_shota_info(trump=trump, qabool=qabool, bid=bid)

    def set_player_status_safe(self, player_index: int, message: str,
                               is_qabool: bool = False, is_first_shooter: bool = False) -> None:
        try:
            self.app.set_player_status(player_index, message,
                                       is_qabool=is_qabool, is_first_shooter=is_first_shooter)
        except TypeError:
            try:
                self.app.set_player_status(player_index, message, is_qabool=is_qabool)
            except TypeError:
                self.app.set_player_status(player_index, message)

    def _explain_card_play(self, player_id: int, card, observation) -> str:
        suit_name = getattr(card.suit, "name", str(card.suit))
        trump_name = getattr(self.trump_suit, "name", "") if self.trump_suit else ""

        leading_suit = None
        if observation.current_trick is not None:
            leading_suit = observation.current_trick.leading_suit

        if leading_suit is None:
            if observation.must_lead_trump:
                return f"Must lead trump ({trump_name})"
            elif card.suit == self.trump_suit:
                return "Leading trump to draw out opponents'"
            elif card.rank.name in ("ACE",):
                return "Leading Ace — guaranteed winner"
            else:
                return f"Leading from {suit_name}"

        leading_suit_name = getattr(leading_suit, "name", str(leading_suit))

        if card.suit == leading_suit:
            suit_cards_in_hand = [c for c in observation.hand if c.suit == leading_suit]
            if len(suit_cards_in_hand) <= 1:
                return f"Only card in {leading_suit_name}"
            if card == min(suit_cards_in_hand, key=lambda c: rank_value(c.rank)):
                return "Playing low — saving high cards"
            elif card == max(suit_cards_in_hand, key=lambda c: rank_value(c.rank)):
                return "Playing high to win the trick"
            else:
                return "Playing just enough to win"

        if card.suit == self.trump_suit:
            return f"Void in {leading_suit_name} — trumping!"
        else:
            return f"Void in {leading_suit_name} — discarding low"
