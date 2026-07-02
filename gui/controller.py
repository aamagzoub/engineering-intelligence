
from agents.random.random_agent import RandomAgent
from environments.wist.environment import WistEnvironment
from environments.wist.playing_engine import PlayingEngine
from environments.wist.round import Round
from environments.wist.setup import create_standard_players
from environments.wist.shota_setup import RandomShotaSetupEngine

class SimulationController:
    """
    Coordinates the GUI and the Wist engine.
    Supports:
    - Auto mode: runs from start to end
    - Step mode: pauses after each phase/trick until Continue is pressed
    """

    def __init__(self, app):
        self.app = app
        self.running = False
        self.paused = False
        self.step_mode = False
        self.phase = "idle"

        self.players = None
        self.round = None
        self.environment = None
        self.agents = None
        self.engine = None

        self.current_trick_number = 0
        self.team_tricks = [0, 0]

        self.trump_suit = None
        self.qabool_player_index = None
        self.bid_value = None
        self.player_bids = ["-", "-", "-", "-"]
        self.first_shooter_player_id = 0

        self.shota_setup_engine = RandomShotaSetupEngine()
        self.shota_setup = None

    # ---------------------------------------------------------
    # Start modes
    # ---------------------------------------------------------

    def start_auto(self) -> None:
        self.step_mode = False
        self.start()

    def start_step_mode(self) -> None:
        self.step_mode = True
        self.start()

    def start(self) -> None:
        if self.running:
            self.app.log("Simulation is already running.")
            return

        self.running = True
        self.paused = False
        self.phase = "dealing"

        self.current_trick_number = 0
        self.team_tricks = [0, 0]

        self.trump_suit = None
        self.qabool_player_index = None
        self.bid_value = None
        self.player_bids = ["-", "-", "-", "-"]
        self.first_shooter_player_id = 0
        self.shota_setup = None

        self.app.clear_log()
        self.app.reset_player_statuses()
        self.app.set_status("Simulation running...")
        self.app.set_current_trick("Dealing cards...")
        self.set_shota_info_safe(
            trump="-",
            qabool="-",
            bid="-",
            first_shooter="-",
        )

        self.app.log("Simulation started.")
        self.app.log(f"Mode: {'Step-by-step' if self.step_mode else 'Auto'}")

        self.players = create_standard_players()

        self.round = Round(self.players)
        self.round.deal()

        self.environment = WistEnvironment(self.round.state)

        self.agents = [
            RandomAgent(),
            RandomAgent(),
            RandomAgent(),
            RandomAgent(),
        ]

        self.engine = PlayingEngine()

        self.app.log("Players created.")
        self.app.log("Cards dealt.")

        self.show_player_hands()

        self.phase = "dealt"

        if self.step_mode:
            self.app.set_status("Paused after dealing. Press Continue.")
            self.app.set_current_trick(
                "Cards dealt.\nReview all hands, then press Continue."
            )
            return

        self.continue_simulation()

    # ---------------------------------------------------------
    # Phase control
    # ---------------------------------------------------------

    def continue_simulation(self) -> None:
        if not self.running:
            self.app.log("No running simulation. Start first.")
            return

        self.paused = False

        if self.phase == "dealt":
            self.prepare_shota_setup()
            self.phase = "setup_ready"

            if self.step_mode:
                self.app.set_status("Paused after bidding/setup. Press Continue.")
                self.app.set_current_trick(
                    "Bidding/setup complete.\n"
                    "Review Qabool, bid, trump and player bids."
                )
                return

        if self.phase == "setup_ready":
            self.phase = "before_first_trick"
            self.first_shooter_player_id = self.round.next_leading_player_id

            self.set_shota_info_safe(
                trump=self.trump_suit.name,
                qabool=f"Player {self.qabool_player_index + 1}",
                bid=str(self.bid_value),
                first_shooter=f"Player {self.first_shooter_player_id + 1}",
            )

            self.app.log("")
            self.app.log("=== Before First Trick ===")
            self.app.log(f"First shooter: Player {self.first_shooter_player_id + 1}")

            for player_index in range(4):
                self.set_player_status_safe(
                    player_index,
                    "Ready",
                    is_qabool=(player_index == self.qabool_player_index),
                    is_first_shooter=(player_index == self.first_shooter_player_id),
                )
                self.app.set_player_bid(player_index, self.player_bids[player_index])

            if self.step_mode:
                self.app.set_status("Paused before first trick. Press Continue.")
                self.app.set_current_trick(
                    f"Ready for first trick.\n"
                    f"First shooter: Player {self.first_shooter_player_id + 1}"
                )
                return

        if self.phase == "before_first_trick":
            self.phase = "playing_tricks"

        if self.phase == "playing_tricks":
            if self.step_mode:
                self.play_next_trick_step()
            else:
                self.app.root.after(700, self.play_next_trick_auto)
            return

        if self.phase == "completed":
            self.app.log("Simulation already completed.")
            return

    def pause(self) -> None:
        if not self.running:
            self.app.log("No running simulation to pause.")
            return

        self.paused = True
        self.app.set_status("Simulation paused.")
        self.app.log("Simulation paused. Press Continue to resume.")

    def stop(self) -> None:
        if not self.running:
            self.app.log("Simulation is not running.")
            return

        self.running = False
        self.paused = False
        self.phase = "stopped"

        self.app.set_status("Simulation stopped.")
        self.app.set_current_trick("Simulation stopped.")
        self.app.log("Simulation stopped.")

    # ---------------------------------------------------------
    # Trick execution
    # ---------------------------------------------------------

    def play_next_trick_auto(self) -> None:
        """
        Auto mode:
        play one trick, then schedule the next trick.
        """

        if not self.running or self.paused:
            return

        self.play_one_trick()

        if self.running and self.phase == "playing_tricks":
            self.app.root.after(700, self.play_next_trick_auto)

    def play_next_trick_step(self) -> None:
        """
        Step mode:
        play one trick, then wait for Continue.
        """

        if not self.running:
            return

        self.play_one_trick()

        if self.running and self.phase == "playing_tricks":
            self.app.set_status("Paused after trick. Press Continue.")

    def play_one_trick(self) -> None:
        """
        Play exactly one trick.
        No automatic scheduling happens here.
        """

        if not self.running:
            return

        if self.current_trick_number >= 13:
            self.finish_simulation()
            return

        self.current_trick_number += 1

        for player_index in range(4):
            self.set_player_status_safe(
                player_index,
                "Waiting",
                is_qabool=(player_index == self.qabool_player_index),
                is_first_shooter=(player_index == self.first_shooter_player_id),
            )
            self.app.set_player_bid(player_index, self.player_bids[player_index])

        self.app.clear_played_cards()
        self.show_player_hands()

        self.app.set_current_trick(f"Playing trick {self.current_trick_number}...")
        self.app.log("")
        self.app.log(f"--- Trick {self.current_trick_number} ---")

        trick_details = self.engine.play_trick_details(
            round_=self.round,
            environment=self.environment,
            agents=self.agents,
        )

        winner = trick_details["winner"]
        leader = trick_details["leader"]
        play_order = trick_details["play_order"]
        trick = trick_details["trick"]

        self.app.log(f"Leader: Player {leader + 1}")
        self.app.log(
            "Play order: "
            + " → ".join(f"Player {player_id + 1}" for player_id in play_order)
        )

        winning_team_index = self.get_team_index(winner)
        self.team_tricks[winning_team_index] += 1

        self.show_player_hands()
        self.log_trick_cards(trick)

        self.app.log(f"Player {winner + 1} won the trick.")
        self.app.log(f"Team 1 tricks: {self.team_tricks[0]}")
        self.app.log(f"Team 2 tricks: {self.team_tricks[1]}")

        self.set_player_status_safe(
            winner,
            "Won last trick",
            is_qabool=(winner == self.qabool_player_index),
            is_first_shooter=(winner == self.first_shooter_player_id),
        )

        self.app.set_current_trick(
            f"Trick {self.current_trick_number} complete\n"
            f"Winner: Player {winner + 1}\n"
            f"Team 1: {self.team_tricks[0]} | Team 2: {self.team_tricks[1]}"
        )

        if self.current_trick_number >= 13:
            self.finish_simulation()

    # ---------------------------------------------------------
    # Shota setup
    # ---------------------------------------------------------

    def prepare_shota_setup(self) -> None:
        """
        Prepare the Shota setup using the Shota setup engine.

        For now this uses RandomShotaSetupEngine.
        Later we will replace it with a real rule-based Wist setup engine.
        """

        self.shota_setup = self.shota_setup_engine.create_setup(self.players)

        self.qabool_player_index = self.shota_setup.qabool_player_id
        self.bid_value = self.shota_setup.bid
        self.trump_suit = self.shota_setup.trump_suit

        self.player_bids = ["Pass", "Pass", "Pass", "Pass"]

        for bid_display in self.shota_setup.player_bids:
            self.player_bids[bid_display.player_id] = bid_display.text

        self.round.state.trump_suit = self.trump_suit
        self.first_shooter_player_id = self.round.next_leading_player_id

        self.set_shota_info_safe(
            trump=self.trump_suit.name,
            qabool=f"Player {self.qabool_player_index + 1}",
            bid=str(self.bid_value),
            first_shooter=f"Player {self.first_shooter_player_id + 1}",
        )

        for player_index, bid_text in enumerate(self.player_bids):
            self.app.set_player_bid(player_index, bid_text)
            self.set_player_status_safe(
                player_index,
                "Qabool / Trump" if player_index == self.qabool_player_index else "Waiting",
                is_qabool=(player_index == self.qabool_player_index),
                is_first_shooter=(player_index == self.first_shooter_player_id),
            )

        self.app.log("")
        self.app.log("=== Shota Setup ===")

        for bid_display in self.shota_setup.player_bids:
            if bid_display.is_qabool:
                self.app.log(
                    f"Player {bid_display.player_id + 1}: "
                    f"Bid {bid_display.text}, Trump {self.trump_suit.name}, Qabool"
                )
            else:
                self.app.log(f"Player {bid_display.player_id + 1}: Pass")

        self.app.log(f"Final Qabool player: Player {self.qabool_player_index + 1}")
        self.app.log(f"Playing team: Team {self.shota_setup.playing_team_id + 1}")
        self.app.log(f"Defending team: Team {self.shota_setup.defending_team_id + 1}")
        self.app.log(f"Final Shota bid: {self.bid_value}")
        self.app.log(f"Final trump suit: {self.trump_suit.name}")

    # ---------------------------------------------------------
    # Finish
    # ---------------------------------------------------------

    def finish_simulation(self) -> None:
        self.running = False
        self.paused = False
        self.phase = "completed"

        self.app.set_status("Simulation completed.")
        self.app.set_current_trick(
            f"Full Shota completed\n"
            f"Team 1 tricks: {self.team_tricks[0]}\n"
            f"Team 2 tricks: {self.team_tricks[1]}"
        )

        self.app.log("")
        self.app.log("=== Full Shota completed ===")
        self.app.log(f"Final Team 1 tricks: {self.team_tricks[0]}")
        self.app.log(f"Final Team 2 tricks: {self.team_tricks[1]}")

        if self.team_tricks[0] > self.team_tricks[1]:
            self.app.log("Team 1 wins the Shota.")
        elif self.team_tricks[1] > self.team_tricks[0]:
            self.app.log("Team 2 wins the Shota.")
        else:
            self.app.log("The Shota ended in a draw.")

    # ---------------------------------------------------------
    # GUI-safe helpers
    # ---------------------------------------------------------

    def set_shota_info_safe(
        self,
        trump: str,
        qabool: str = "-",
        bid: str = "-",
        first_shooter: str = "-",
    ) -> None:
        """
        Works with both old and new app.py versions.
        """

        try:
            self.app.set_shota_info(
                trump=trump,
                qabool=qabool,
                bid=bid,
                first_shooter=first_shooter,
            )
        except TypeError:
            self.app.set_shota_info(
                trump=trump,
                qabool=qabool,
                bid=bid,
            )

    def set_player_status_safe(
        self,
        player_index: int,
        message: str,
        is_qabool: bool = False,
        is_first_shooter: bool = False,
    ) -> None:
        """
        Works with both old and new app.py versions.
        """

        try:
            self.app.set_player_status(
                player_index,
                message,
                is_qabool=is_qabool,
                is_first_shooter=is_first_shooter,
            )
        except TypeError:
            try:
                self.app.set_player_status(
                    player_index,
                    message,
                    is_qabool=is_qabool,
                )
            except TypeError:
                self.app.set_player_status(
                    player_index,
                    message,
                )

    # ---------------------------------------------------------
    # Card and trick display helpers
    # ---------------------------------------------------------

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
            self.app.log("Played cards: could not read from Trick model yet.")
            self.app.log(f"Trick internal data: {vars(trick)}")
            return

        self.app.log("Played cards:")

        trick_lines = []
        visual_played_cards = []

        for player_id, card in played_cards:
            full_card_text = self.format_card(card)
            short_card_text = self.format_card_short(card)

            line = f"Player {player_id + 1}: {full_card_text}"
            trick_lines.append(line)
            visual_played_cards.append((player_id, short_card_text))

            self.app.log(line)

        self.app.set_played_cards(visual_played_cards)

    def extract_played_cards_from_trick(self, trick):
        possible_attributes = [
            "played_cards",
            "cards_played",
            "plays",
            "cards",
            "actions",
        ]

        for attribute_name in possible_attributes:
            if not hasattr(trick, attribute_name):
                continue

            value = getattr(trick, attribute_name)

            if isinstance(value, dict):
                return [
                    (player_id, card)
                    for player_id, card in value.items()
                ]

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
        """
        Sort cards by suit, then rank from big to small.

        Suit order:
        Spades, Hearts, Diamonds, Clubs

        Rank order:
        Ace, King, Queen, Jack, 10 ... 2
        """

        suit_order = {
            "SPADES": 0,
            "HEARTS": 1,
            "DIAMONDS": 2,
            "CLUBS": 3,
        }

        rank_order = {
            "ACE": 0,
            "KING": 1,
            "QUEEN": 2,
            "JACK": 3,
            "TEN": 4,
            "NINE": 5,
            "EIGHT": 6,
            "SEVEN": 7,
            "SIX": 8,
            "FIVE": 9,
            "FOUR": 10,
            "THREE": 11,
            "TWO": 12,
        }

        def card_key(card):
            suit = getattr(card, "suit", None)
            rank = getattr(card, "rank", None)

            suit_name = getattr(suit, "name", str(suit)).upper()
            rank_name = getattr(rank, "name", str(rank)).upper()

            return (
                suit_order.get(suit_name, 99),
                rank_order.get(rank_name, 99),
            )

        return sorted(cards, key=card_key)

    def format_card(self, card) -> str:
        rank = getattr(card, "rank", None)
        suit = getattr(card, "suit", None)

        rank_text = getattr(rank, "name", str(rank))
        suit_text = getattr(suit, "name", str(suit))

        return f"{rank_text} of {suit_text}"

    def format_card_short(self, card) -> str:
        rank = getattr(card, "rank", None)
        suit = getattr(card, "suit", None)

        rank_name = getattr(rank, "name", str(rank)).upper()
        suit_name = getattr(suit, "name", str(suit)).upper()

        rank_map = {
            "ACE": "A",
            "KING": "K",
            "QUEEN": "Q",
            "JACK": "J",
            "TEN": "10",
            "NINE": "9",
            "EIGHT": "8",
            "SEVEN": "7",
            "SIX": "6",
            "FIVE": "5",
            "FOUR": "4",
            "THREE": "3",
            "TWO": "2",
        }

        suit_map = {
            "SPADES": "♠",
            "HEARTS": "♥",
            "DIAMONDS": "♦",
            "CLUBS": "♣",
        }

        rank_text = rank_map.get(rank_name, rank_name[:1])
        suit_text = suit_map.get(suit_name, "?")

        return f"{rank_text}{suit_text}"

    def get_team_index(self, player_index: int) -> int:
        if player_index in [0, 2]:
            return 0

        return 1