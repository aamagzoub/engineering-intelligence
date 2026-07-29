"""
Rule-based Wist agent.

Implements basic but sound Wist strategy heuristics for both
bidding and card play. This is NOT a learning agent — it uses
fixed rules derived from how experienced Wist players think.

Strategy principles:
- Bidding: bid based on hand strength (long suits + high cards)
- Leading: lead trump to extract opponents' trumps, then side winners
- Following: play lowest when can't win, play just enough to win
- Trumping: trump when void in led suit and partner isn't winning
- Discarding: throw lowest card from weakest suit when void
"""

from collections import Counter

from environments.wist.actions import BidAction, PassAction, PlayCardAction
from environments.wist.observation import BiddingObservation, WistObservation
from environments.wist.rules import legal_cards, rank_value
from intelligence.core.action import Action
from intelligence.core.agent import Agent
from intelligence.core.cards.card import Card
from intelligence.core.cards.rank import Rank
from intelligence.core.cards.suit import Suit
from intelligence.core.observation import Observation


# High cards that are likely to win tricks on their own.
HIGH_RANKS = {Rank.ACE, Rank.KING, Rank.QUEEN}

# Rank values for trick estimation.
RANK_VALUES_FOR_ESTIMATE = {
    Rank.TWO: 2, Rank.THREE: 3, Rank.FOUR: 4, Rank.FIVE: 5,
    Rank.SIX: 6, Rank.SEVEN: 7, Rank.EIGHT: 8, Rank.NINE: 9,
    Rank.TEN: 10, Rank.JACK: 11, Rank.QUEEN: 12, Rank.KING: 13, Rank.ACE: 14,
}


class RuleBasedAgent(Agent):
    """
    A heuristic Wist agent that plays using encoded strategy rules.
    No learning — pure hand-crafted logic.
    """

    def act(self, observation: Observation) -> Action:
        if isinstance(observation, BiddingObservation):
            return self._act_bidding(observation)

        if isinstance(observation, WistObservation):
            return self._act_play(observation)

        raise TypeError(
            f"RuleBasedAgent does not support {type(observation).__name__}."
        )

    # ==========================================================
    # BIDDING STRATEGY
    # ==========================================================

    def _act_bidding(self, obs: BiddingObservation) -> Action:
        """
        Bidding strategy:
        1. Determine the mandatory bid value (longest suit + 3).
        2. Estimate hand strength to decide whether to bid or pass.
        3. The bid NUMBER is fixed — only the decision to bid is strategic.
        """
        hand = obs.hand
        suit_counts = Counter(card.suit for card in hand)
        longest_suit_count = max(suit_counts.values()) if suit_counts else 0

        # Cannot bid with 8+ in one suit (Dak).
        if longest_suit_count >= 8:
            return PassAction(player_id=obs.player_id)

        # The bid value is fixed by longest suit count.
        bid_value = longest_suit_count + 3  # Standard formula.
        bid_value = max(7, min(bid_value, 13))

        # Estimate hand strength (decides whether to bid, not what to bid).
        expected_tricks = self._estimate_tricks(hand, suit_counts)

        if obs.is_sahib_al_qabool:
            return self._bid_as_qabool(obs, bid_value, expected_tricks)
        else:
            return self._bid_as_regular(obs, bid_value, expected_tricks)

    def _estimate_tricks(self, hand: list[Card], suit_counts: Counter) -> int:
        """
        Estimate how many tricks this hand can win.

        The standard Wist formula: longest suit count + 3 = max bid.
        But the EXPECTED tricks is closer to longest suit count itself
        (your trumps will likely win) plus side-suit winners.

        Simplified heuristic:
        - Each trump card with rank >= 10 (10, J, Q, K, A): likely wins = 1
        - Each trump card below 10 in a long suit (5+): ruffing potential = 0.5
        - Side suit Ace: 0.9
        - Side suit King with 2+ cards in suit: 0.5
        """
        if not hand:
            return 0

        longest_suit = max(suit_counts, key=suit_counts.get)
        longest_count = suit_counts[longest_suit]

        tricks = 0.0

        for card in hand:
            if card.suit == longest_suit:
                # Trump suit — high cards are near-certain winners.
                rv = RANK_VALUES_FOR_ESTIMATE.get(card.rank, 0)
                if rv >= 10:
                    tricks += 1.0
                elif longest_count >= 5:
                    tricks += 0.5  # Low trump in long suit can ruff.
            else:
                # Side suit.
                suit_len = suit_counts[card.suit]
                if card.rank == Rank.ACE:
                    tricks += 0.9
                elif card.rank == Rank.KING and suit_len >= 2:
                    tricks += 0.5

        return int(tricks)

    def _bid_as_regular(self, obs: BiddingObservation, bid_value: int,
                        expected_tricks: int) -> Action:
        """
        Regular player bidding logic.

        The bid value is FIXED by longest suit: longest + 3.
        The only decision is: bid or pass.
        Decision to bid is based on hand strength (high cards, partner hope).
        """

        # The bid value is determined by longest suit — not a choice.
        # bid_value is already calculated as longest_suit + 3.

        # Decide whether to bid based on hand strength.
        if expected_tricks < 5:
            return PassAction(player_id=obs.player_id)

        # Opening bid cannot exceed 11.
        if obs.is_opening_bid and bid_value > 11:
            return PassAction(player_id=obs.player_id)

        # Must beat current highest bid.
        if obs.current_highest_bid is not None:
            if bid_value <= obs.current_highest_bid:
                # Our mandatory bid can't beat it — pass.
                return PassAction(player_id=obs.player_id)

        # Validate range.
        if bid_value < 7 or bid_value > 13:
            return PassAction(player_id=obs.player_id)

        return BidAction(player_id=obs.player_id, value=bid_value)

    def _bid_as_qabool(self, obs: BiddingObservation, bid_value: int,
                       expected_tricks: int) -> Action:
        """
        Sahib Al-Qabool bidding logic.

        Qabool's special rules:
        - Can match (not required to go higher).
        - Both bid-limit restrictions are lifted (can bid anything).
        - If all passed: can use "extra card advantage" (bid one lower).
        - If all passed and decides not to play: declares Dak.
        """

        current_bid = obs.current_highest_bid
        all_passed = (current_bid is None)

        if all_passed:
            # Everyone passed — Qabool can bid or declare Dak.
            # Extra card advantage: can bid one lower than standard formula.
            # Standard: longest+3. With advantage: longest+2. Min is 7.
            if expected_tricks >= 3:
                advantage_bid = max(7, bid_value - 1)
                return BidAction(player_id=obs.player_id, value=advantage_bid)
            # Truly weak — declare Dak.
            return PassAction(player_id=obs.player_id)

        # Someone bid. Accept (pass) or match/outbid.
        # Qabool can match — bid the same number.
        if expected_tricks >= current_bid - 2:
            # Strong enough to contest. Match or outbid.
            match_bid = max(current_bid, bid_value)
            match_bid = min(match_bid, 13)
            return BidAction(player_id=obs.player_id, value=match_bid)

        # Can't compete — accept their bid (pass).
        return PassAction(player_id=obs.player_id)

    # ==========================================================
    # CARD PLAY STRATEGY
    # ==========================================================

    def _act_play(self, obs: WistObservation) -> Action:
        """
        Card play strategy dispatcher.
        """
        if not obs.hand:
            raise ValueError("RuleBasedAgent cannot act with an empty hand.")

        leading_suit = None
        if obs.current_trick is not None:
            leading_suit = obs.current_trick.leading_suit

        must_lead_trump = None
        if obs.must_lead_trump and obs.trump_suit is not None:
            must_lead_trump = obs.trump_suit

        playable = legal_cards(
            hand=obs.hand,
            leading_suit=leading_suit,
            must_lead_trump=must_lead_trump,
        )

        # If only one legal card, play it.
        if len(playable) == 1:
            return PlayCardAction(player_id=obs.player_id, card=playable[0])

        # Determine our position in the trick.
        if obs.current_trick is None or len(obs.current_trick.played_cards) == 0:
            # We're leading.
            card = self._choose_lead(obs, playable)
        else:
            # We're following.
            card = self._choose_follow(obs, playable, leading_suit)

        return PlayCardAction(player_id=obs.player_id, card=card)

    # ----------------------------------------------------------
    # Leading strategy
    # ----------------------------------------------------------

    def _choose_lead(self, obs: WistObservation, playable: list[Card]) -> Card:
        """
        Choose which card to lead with.

        Strategy:
        1. If must lead trump → lead highest trump.
        2. If we have lots of trumps → lead trump to draw them out.
        3. Otherwise lead an Ace from a side suit (guaranteed winner).
        4. Otherwise lead from our longest side suit.
        5. Fallback: lead lowest card.
        """
        trump = obs.trump_suit

        # Must lead trump (first trick rule).
        if obs.must_lead_trump:
            return self._highest_card(playable)

        # Count our trumps.
        trump_cards = [c for c in obs.hand if c.suit == trump]

        # If we have 4+ trumps, lead trump to extract opponents' trumps.
        if len(trump_cards) >= 4:
            trump_playable = [c for c in playable if c.suit == trump]
            if trump_playable:
                return self._highest_card(trump_playable)

        # Lead an Ace from a side suit (guaranteed winner if no trump).
        side_aces = [c for c in playable if c.rank == Rank.ACE and c.suit != trump]
        if side_aces:
            return side_aces[0]

        # Lead from longest side suit (to establish long cards).
        suit_counts = Counter(c.suit for c in obs.hand)
        side_suits = [(s, cnt) for s, cnt in suit_counts.items()
                      if s != trump and cnt > 0]
        if side_suits:
            side_suits.sort(key=lambda x: -x[1])
            best_side_suit = side_suits[0][0]
            suit_cards = [c for c in playable if c.suit == best_side_suit]
            if suit_cards:
                # Lead highest from longest side suit.
                return self._highest_card(suit_cards)

        # Fallback: lead lowest card.
        return self._lowest_card(playable)

    # ----------------------------------------------------------
    # Following strategy
    # ----------------------------------------------------------

    def _choose_follow(self, obs: WistObservation, playable: list[Card],
                       leading_suit: Suit | None) -> Card:
        """
        Choose which card to play when following.

        Strategy depends on whether we're following suit, trumping, or discarding.
        """
        trump = obs.trump_suit
        trick = obs.current_trick
        partner_id = self._partner_id(obs.player_id)

        # Determine current winning card in the trick.
        current_winner_id, current_winner_card = self._current_trick_winner(
            trick, trump
        )

        # Is our partner currently winning?
        partner_winning = (current_winner_id == partner_id)

        # Case 1: We have cards in the leading suit (following suit).
        if leading_suit and playable[0].suit == leading_suit:
            return self._follow_suit(
                playable, current_winner_card, partner_winning, trump
            )

        # Case 2: We're void in the leading suit.
        # Can we trump?
        trump_cards = [c for c in playable if c.suit == trump]

        if trump_cards and not partner_winning:
            # Trump with the lowest trump that wins.
            current_trump_in_trick = self._highest_trump_in_trick(trick, trump)
            if current_trump_in_trick is not None:
                # Need to beat existing trump.
                winning_trumps = [c for c in trump_cards
                                  if rank_value(c.rank) > rank_value(current_trump_in_trick.rank)]
                if winning_trumps:
                    return self._lowest_card(winning_trumps)
            else:
                # No trump in trick yet — any trump wins.
                return self._lowest_card(trump_cards)

        # Case 3: Partner is winning or we can't trump effectively.
        # Discard lowest from weakest suit.
        return self._best_discard(playable, trump, obs.hand)

    def _follow_suit(self, playable: list[Card], current_winner_card: Card | None,
                     partner_winning: bool, trump: Suit | None) -> Card:
        """
        Follow suit strategy.

        - If partner is winning: play lowest (don't waste high cards).
        - If we can beat the current winner: play the lowest card that wins.
        - If we can't beat it: play lowest (save high cards for later).
        """
        if partner_winning:
            return self._lowest_card(playable)

        if current_winner_card is None:
            # We're second to play — play highest to try to win.
            return self._highest_card(playable)

        # Can we beat the current winner?
        # Only matters if winner is in the same suit (not a trump).
        if current_winner_card.suit == playable[0].suit:
            winners = [c for c in playable
                       if rank_value(c.rank) > rank_value(current_winner_card.rank)]
            if winners:
                # Play the lowest card that still wins.
                return self._lowest_card(winners)

        # Can't beat it — play lowest to save high cards.
        return self._lowest_card(playable)

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    def _partner_id(self, player_id: int) -> int:
        """Partner is the player sitting opposite (offset +2)."""
        return (player_id + 2) % 4

    def _current_trick_winner(self, trick, trump: Suit | None):
        """
        Determine who is currently winning the trick (so far).
        Returns (player_id, card) or (None, None) if trick is empty.
        """
        if trick is None or not trick.played_cards:
            return None, None

        leading_suit = trick.leading_suit
        best_id = trick.played_cards[0].player_id
        best_card = trick.played_cards[0].card

        for played in trick.played_cards[1:]:
            card = played.card

            # Trump beats everything non-trump.
            if card.suit == trump and best_card.suit != trump:
                best_id = played.player_id
                best_card = card
            elif card.suit == trump and best_card.suit == trump:
                if rank_value(card.rank) > rank_value(best_card.rank):
                    best_id = played.player_id
                    best_card = card
            elif card.suit == leading_suit and best_card.suit == leading_suit:
                if rank_value(card.rank) > rank_value(best_card.rank):
                    best_id = played.player_id
                    best_card = card
            # Off-suit non-trump never wins.

        return best_id, best_card

    def _highest_trump_in_trick(self, trick, trump: Suit | None) -> Card | None:
        """Return the highest trump card played in the trick so far."""
        if trick is None or not trick.played_cards or trump is None:
            return None

        trump_cards = [p.card for p in trick.played_cards if p.card.suit == trump]
        if not trump_cards:
            return None

        return max(trump_cards, key=lambda c: rank_value(c.rank))

    def _best_discard(self, playable: list[Card], trump: Suit | None,
                      full_hand: list[Card]) -> Card:
        """
        Choose the best card to discard when void in led suit
        and not trumping.

        Strategy: discard lowest card from the suit with fewest cards
        (weakest suit — least chance of winning tricks there anyway).
        """
        # Prefer discarding non-trump low cards.
        non_trump = [c for c in playable if c.suit != trump]
        if non_trump:
            # From non-trump playable, find the suit with fewest remaining cards.
            suit_counts = Counter(c.suit for c in full_hand if c.suit != trump)
            # Sort by suit count (ascending), then by rank (ascending).
            non_trump.sort(key=lambda c: (suit_counts.get(c.suit, 0), rank_value(c.rank)))
            return non_trump[0]

        # Only trump cards available — discard lowest trump.
        return self._lowest_card(playable)

    def _highest_card(self, cards: list[Card]) -> Card:
        """Return the highest-ranked card."""
        return max(cards, key=lambda c: rank_value(c.rank))

    def _lowest_card(self, cards: list[Card]) -> Card:
        """Return the lowest-ranked card."""
        return min(cards, key=lambda c: rank_value(c.rank))
