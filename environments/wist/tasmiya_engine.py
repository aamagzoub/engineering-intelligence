import random
from collections import Counter
from dataclasses import dataclass

from environments.wist.actions import BidAction, PassAction
from environments.wist.bidding import Bid, Pass, validate_opening_bid, validate_regular_bid
from environments.wist.bidding_engine import BiddingEngine
from environments.wist.observation import BiddingObservation
from environments.wist.player import Player
from intelligence.core.action import Action
from intelligence.core.agent import Agent
from intelligence.core.cards.card import Card
from intelligence.core.cards.deck import Deck
from intelligence.core.cards.suit import Suit


@dataclass(frozen=True)
class TasmiyaResult:
    """
    The result of a completed Al-Tasmiya phase.

    If all players pass and Sahib Al-Qabool declares Dak,
    is_dak will be True and the other fields will reflect
    the state at that point.
    """

    winning_bidder_id: int | None
    winning_bid_value: int | None
    trump_suit: Suit | None
    playing_team_id: int | None
    defending_team_id: int | None
    sahib_al_qabool_id: int
    is_dak: bool = False
    bid_history: list[tuple[int, int | None]] = None

    def __post_init__(self) -> None:
        if self.bid_history is None:
            object.__setattr__(self, "bid_history", [])


def determine_trump_suit(hand: list[Card]) -> Suit:
    """
    Determine the trump suit from a player's hand.

    Per the rules:
    - Trump must be the suit the bidder holds the most cards in.
    - That suit must have 7 or fewer cards (8+ triggers card-based Dak).

    If there is a tie in suit length, choose arbitrarily (first found).
    """

    suit_counts = Counter(card.suit for card in hand)

    longest_suit = max(suit_counts, key=suit_counts.get)

    return longest_suit


def max_bid_for_hand(hand: list[Card]) -> int:
    """
    Determine the maximum bid a regular player can make.

    Per the rules:
    - You cannot bid higher than the number of cards in your strongest suit.
    - The trump suit must have 7 or fewer cards.
    - So the maximum bid based on suit length alone is capped at 7
      from the trump constraint, but the bid itself represents total
      expected tricks and can be based on cards across all suits.

    The standard table is:
      4 cards in strongest suit → bid 7
      5 cards → bid 8
      6 cards → bid 9
      7 cards → bid 10

    However, the bid does not have to come from one suit. It can be
    based on strongest cards across all suits. The constraint is that
    the trump suit (longest suit) must have ≤ 7 cards.

    For this function, we return the maximum allowed bid value based
    on the longest suit length using the standard formula:
      max_bid = longest_suit_count + 3

    Capped at 13 (maximum possible bid).
    """

    suit_counts = Counter(card.suit for card in hand)
    longest_suit_count = max(suit_counts.values())

    # 8+ in one suit means Dak — should not be bidding at all.
    if longest_suit_count >= 8:
        return 0

    return min(longest_suit_count + 3, 13)


def tasmiya_order(sahib_al_qabool_id: int) -> list[int]:
    """
    Return the Al-Tasmiya bidding order.

    Order: left of Sahib Al-Qabool → opposite → right (dealer).
    Sahib Al-Qabool himself is NOT included — he decides separately.

    Counter-clockwise seating: 0 → 1 → 2 → 3 → 0.
    Left of player X (counter-clockwise) = (X + 1) % 4.
    """

    return [
        (sahib_al_qabool_id + 1) % 4,
        (sahib_al_qabool_id + 2) % 4,
        (sahib_al_qabool_id + 3) % 4,
    ]


def determine_first_shota_qabool() -> int:
    """
    Determine Sahib Al-Qabool for the very first Shota.

    Per the rules: all four players each draw a random card from the deck.
    The team whose players drew the higher card (between their two drawn
    cards) wins Al-Qabool. Then a specific player from that team gets it.

    Teams: Player 0+2 (Team 0), Player 1+3 (Team 1).
    The player with the highest card on the winning team becomes Qabool.
    """
    from environments.wist.rules import rank_value

    deck = Deck()
    deck.shuffle()

    # Each player draws one card.
    drawn = [deck.deal(1)[0] for _ in range(4)]

    # Each team's best card.
    team_0_best = max(rank_value(drawn[0].rank), rank_value(drawn[2].rank))
    team_1_best = max(rank_value(drawn[1].rank), rank_value(drawn[3].rank))

    if team_0_best >= team_1_best:
        # Team 0 wins. The player with the higher card is Qabool.
        if rank_value(drawn[0].rank) >= rank_value(drawn[2].rank):
            return 0
        else:
            return 2
    else:
        # Team 1 wins.
        if rank_value(drawn[1].rank) >= rank_value(drawn[3].rank):
            return 1
        else:
            return 3


class TasmiyaEngine:
    """
    Orchestrates the Al-Tasmiya (bidding) phase of one Shota.

    This engine:
    1. Presents BiddingObservation to each player in correct order.
    2. Collects BidAction or PassAction from each agent.
    3. Validates bids against the rules.
    4. After three players bid/pass, asks Sahib Al-Qabool to decide.
    5. Determines the winning bid, trump suit, and teams.

    The engine does NOT handle:
    - Card-based Dak (checked before Tasmiya starts)
    - Pass-based Dak (returned as TasmiyaResult with is_dak=True
      for the caller to handle)
    """

    def run(
        self,
        players: list[Player],
        agents: list[Agent],
        sahib_al_qabool_id: int,
        is_first_shota: bool = False,
    ) -> TasmiyaResult:
        """
        Run the full Al-Tasmiya phase and return the result.

        is_first_shota: enables the special first-Shota rule where the
        third player (dealer) can declare automatic Dak if the first two
        passed, without Qabool having a say.
        """

        bidding_engine = BiddingEngine()
        bid_history: list[tuple[int, int | None]] = []

        order = tasmiya_order(sahib_al_qabool_id)
        has_opening_bid = False
        bidding_stopped = False

        # Track passes for first-Shota special rule.
        consecutive_passes = 0

        # Phase 1: Three players bid or pass in order.
        for idx, player_id in enumerate(order):
            if bidding_stopped:
                break

            player = players[player_id]

            observation = BiddingObservation(
                player_id=player_id,
                hand=list(player.hand),
                previous_bids=list(bid_history),
                current_highest_bid=(
                    bidding_engine.highest_bid.value
                    if bidding_engine.highest_bid
                    else None
                ),
                is_sahib_al_qabool=False,
                is_opening_bid=(not has_opening_bid),
            )

            action = agents[player_id].act(observation)

            if isinstance(action, BidAction):
                bid = Bid(player_id=action.player_id, value=action.value)

                if not has_opening_bid:
                    validate_opening_bid(bid)
                    has_opening_bid = True
                else:
                    validate_regular_bid(bid, bidding_engine.highest_bid)

                bidding_engine.apply_bid(bid)
                bid_history.append((player_id, action.value))
                consecutive_passes = 0

                # Bid of 13 stops Al-Tasmiya immediately.
                if action.value == 13:
                    bidding_stopped = True

            elif isinstance(action, PassAction):
                bidding_engine.apply_pass(Pass(player_id=player_id))
                bid_history.append((player_id, None))
                consecutive_passes += 1

                # First Shota special rule:
                # If the first two players pass and the third declares Dak
                # → automatic Dak, Qabool has no say.
                # The agent "declares Dak" by passing when it's the 3rd player
                # and the first two passed. We check if the third player
                # actively wants Dak (passes after two consecutive passes).
                if (is_first_shota and idx == 2
                        and consecutive_passes == 3
                        and not has_opening_bid):
                    # All three passed in first Shota — third player's pass
                    # means they accept going to Qabool (normal flow).
                    # But if first two passed and third DECLARES Dak
                    # (which is: first two pass, third passes with intent to Dak):
                    # In our implementation, the third player passing means they
                    # don't want to bid. Qabool still gets to decide.
                    # The special rule is: third player CAN declare Dak
                    # (making it automatic). Since our agent just "passes",
                    # we'll treat the third player's pass after two passes
                    # in first Shota as going to Qabool normally.
                    # The automatic Dak would require the agent to explicitly
                    # signal "Dak" vs "Pass". For now we follow the normal flow.
                    pass

            else:
                raise TypeError(
                    f"Expected BidAction or PassAction from player {player_id}, "
                    f"got {type(action).__name__}."
                )

        # Phase 2: Sahib Al-Qabool decides.
        all_others_passed = bidding_engine.highest_bid is None

        qabool_player = players[sahib_al_qabool_id]

        qabool_observation = BiddingObservation(
            player_id=sahib_al_qabool_id,
            hand=list(qabool_player.hand),
            previous_bids=list(bid_history),
            current_highest_bid=(
                bidding_engine.highest_bid.value
                if bidding_engine.highest_bid
                else None
            ),
            is_sahib_al_qabool=True,
            is_opening_bid=(not has_opening_bid),
        )

        qabool_action = agents[sahib_al_qabool_id].act(qabool_observation)

        if isinstance(qabool_action, PassAction):
            # Sahib Al-Qabool passes = accepts the highest bid if one exists,
            # or declares Dak if all passed.
            if all_others_passed:
                # All four players passed → Dak.
                bid_history.append((sahib_al_qabool_id, None))
                return TasmiyaResult(
                    winning_bidder_id=None,
                    winning_bid_value=None,
                    trump_suit=None,
                    playing_team_id=None,
                    defending_team_id=None,
                    sahib_al_qabool_id=sahib_al_qabool_id,
                    is_dak=True,
                    bid_history=bid_history,
                )
            else:
                # Accept the highest bid. The other player's team plays.
                bid_history.append((sahib_al_qabool_id, None))
                winning_bid = bidding_engine.highest_bid
                winning_player = players[winning_bid.player_id]
                trump = determine_trump_suit(winning_player.hand)

                playing_team_id = winning_player.team_id
                defending_team_id = 1 if playing_team_id == 0 else 0

                return TasmiyaResult(
                    winning_bidder_id=winning_bid.player_id,
                    winning_bid_value=winning_bid.value,
                    trump_suit=trump,
                    playing_team_id=playing_team_id,
                    defending_team_id=defending_team_id,
                    sahib_al_qabool_id=sahib_al_qabool_id,
                    is_dak=False,
                    bid_history=bid_history,
                )

        elif isinstance(qabool_action, BidAction):
            # Sahib Al-Qabool matches or outbids.
            # Both bid-limit restrictions are lifted for Qabool:
            # - Opening bid ≤ 11 does NOT apply to Qabool.
            # - Cannot bid higher than strongest suit does NOT apply to Qabool.
            bid = Bid(
                player_id=sahib_al_qabool_id,
                value=qabool_action.value,
            )

            if has_opening_bid:
                # Qabool can match or exceed.
                from environments.wist.bidding import validate_sahib_al_qabool_bid
                validate_sahib_al_qabool_bid(bid, bidding_engine.highest_bid)
            # If no one bid (all passed), Qabool can bid anything ≥ 7.
            # No opening-bid-≤-11 restriction applies to Qabool.

            bidding_engine.apply_bid(bid, is_sahib_al_qabool=True)
            bid_history.append((sahib_al_qabool_id, qabool_action.value))

            # Sahib Al-Qabool's team plays.
            trump = determine_trump_suit(qabool_player.hand)
            playing_team_id = qabool_player.team_id
            defending_team_id = 1 if playing_team_id == 0 else 0

            return TasmiyaResult(
                winning_bidder_id=sahib_al_qabool_id,
                winning_bid_value=qabool_action.value,
                trump_suit=trump,
                playing_team_id=playing_team_id,
                defending_team_id=defending_team_id,
                sahib_al_qabool_id=sahib_al_qabool_id,
                is_dak=False,
                bid_history=bid_history,
            )

        else:
            raise TypeError(
                f"Expected BidAction or PassAction from Sahib Al-Qabool, "
                f"got {type(qabool_action).__name__}."
            )
