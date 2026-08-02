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
    #
    # Goal hierarchy:
    # 1. WIN ALL 13 TRICKS (Seek) — always play to maximize this chance
    # 2. If Seek is lost, maximize your own tricks won
    # 3. If you can't win many, minimize opponent's tricks
    # ==========================================================

    def _act_play(self, obs: WistObservation) -> Action:
        """Card play strategy — always play to win every trick."""
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

        if len(playable) == 1:
            return PlayCardAction(player_id=obs.player_id, card=playable[0])

        if obs.current_trick is None or len(obs.current_trick.played_cards) == 0:
            card = self._choose_lead(obs, playable)
        else:
            card = self._choose_follow(obs, playable, leading_suit)

        return PlayCardAction(player_id=obs.player_id, card=card)

    # ----------------------------------------------------------
    # Leading strategy — aggressive, Seek-oriented
    # ----------------------------------------------------------

    def _choose_lead(self, obs: WistObservation, playable: list[Card]) -> Card:
        """
        Leading strategy (Seek-first mindset):
        1. Must lead trump (first trick) → highest trump.
        2. Lead trump to flush out opponents' trumps (dominate trump suit).
        3. Once trumps are likely exhausted, lead side-suit Aces/Kings (guaranteed wins).
        4. Lead from longest side suit to establish long cards.
        """
        trump = obs.trump_suit

        # Must lead trump (first trick rule).
        if obs.must_lead_trump:
            return self._highest_card(playable)

        # Count trumps in hand.
        trump_in_hand = [c for c in obs.hand if c.suit == trump]
        trump_playable = [c for c in playable if c.suit == trump]

        # Aggressively lead trump while we have them — flush out opponents' trumps.
        # This is the Seek strategy: control the trump suit entirely.
        if trump_playable and len(trump_in_hand) >= 2:
            return self._highest_card(trump_playable)

        # If we have one trump left, save it for later (ruffing opportunity).
        # Lead guaranteed side-suit winners instead.

        # Lead Aces (guaranteed winners when no trump is out).
        side_aces = [c for c in playable if c.rank == Rank.ACE and c.suit != trump]
        if side_aces:
            # Lead from the shortest side suit first (clear it for future voids).
            suit_counts = Counter(c.suit for c in obs.hand)
            side_aces.sort(key=lambda c: suit_counts.get(c.suit, 0))
            return side_aces[0]

        # Lead Kings from suits where Ace is already played (likely winners).
        side_kings = [c for c in playable if c.rank == Rank.KING and c.suit != trump]
        if side_kings:
            return side_kings[0]

        # Lead from longest side suit — high card to try to win.
        suit_counts = Counter(c.suit for c in obs.hand)
        side_suits = [(s, cnt) for s, cnt in suit_counts.items() if s != trump and cnt > 0]
        if side_suits:
            side_suits.sort(key=lambda x: -x[1])
            best_side = side_suits[0][0]
            suit_cards = [c for c in playable if c.suit == best_side]
            if suit_cards:
                return self._highest_card(suit_cards)

        # Last resort — lead highest remaining trump.
        if trump_playable:
            return self._highest_card(trump_playable)

        return self._highest_card(playable)

    # ----------------------------------------------------------
    # Following strategy — always try to WIN the trick
    # ----------------------------------------------------------

    def _choose_follow(self, obs: WistObservation, playable: list[Card],
                       leading_suit: Suit | None) -> Card:
        """
        Following strategy (Seek mindset — win every trick):
        - If we can win the trick, play the card that wins.
        - Only concede if partner is safely winning AND we're 4th to play.
        - Trump aggressively when void in led suit.
        """
        trump = obs.trump_suit
        trick = obs.current_trick
        partner_id = self._partner_id(obs.player_id)

        current_winner_id, current_winner_card = self._current_trick_winner(trick, trump)
        partner_winning = (current_winner_id == partner_id)
        cards_played_in_trick = len(trick.played_cards) if trick else 0

        # Case 1: Following suit.
        if leading_suit and playable[0].suit == leading_suit:
            # Partner is winning AND they have the highest possible card AND we're last.
            if partner_winning and cards_played_in_trick == 3:
                return self._lowest_card(playable)

            # Try to win — play highest card that beats the current winner.
            if current_winner_card and current_winner_card.suit == leading_suit:
                winners = [c for c in playable
                           if rank_value(c.rank) > rank_value(current_winner_card.rank)]
                if winners:
                    # Play highest to secure the trick (Seek mindset).
                    return self._highest_card(winners)

            # No current winner in our suit (they trumped) — play lowest.
            if current_winner_card and current_winner_card.suit != leading_suit:
                return self._lowest_card(playable)

            # We're early in the trick or no one winning yet — play highest.
            return self._highest_card(playable)

        # Case 2: Void in led suit — TRUMP aggressively.
        trump_cards = [c for c in playable if c.suit == trump]

        if trump_cards:
            # Partner is winning safely and we're last — don't over-trump partner.
            if partner_winning and cards_played_in_trick == 3:
                return self._best_discard(playable, trump, obs.hand)

            # Trump to win — use highest trump to guarantee we take it.
            current_trump = self._highest_trump_in_trick(trick, trump)
            if current_trump is not None:
                # Need to beat existing trump.
                beating = [c for c in trump_cards
                           if rank_value(c.rank) > rank_value(current_trump.rank)]
                if beating:
                    return self._highest_card(beating)
                # Can't beat their trump — discard instead.
                return self._best_discard(playable, trump, obs.hand)
            else:
                # No trump in trick — our trump wins for sure. Use highest.
                return self._highest_card(trump_cards)

        # Case 3: Can't follow suit and can't trump — discard strategically.
        return self._best_discard(playable, trump, obs.hand)

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
        Discard strategy (Seek-defensive):
        - Throw from shortest non-trump suit to create voids faster.
        - Voids let us trump in the future (win more tricks).
        - Always discard the lowest card from the weakest suit.
        """
        non_trump = [c for c in playable if c.suit != trump]
        if non_trump:
            # Discard from the suit with fewest remaining cards (create void).
            suit_counts = Counter(c.suit for c in full_hand if c.suit != trump)
            non_trump.sort(key=lambda c: (suit_counts.get(c.suit, 0), rank_value(c.rank)))
            return non_trump[0]

        # Only trump available — discard lowest trump.
        return self._lowest_card(playable)

    def _highest_card(self, cards: list[Card]) -> Card:
        """Return the highest-ranked card."""
        return max(cards, key=lambda c: rank_value(c.rank))

    def _lowest_card(self, cards: list[Card]) -> Card:
        """Return the lowest-ranked card."""
        return min(cards, key=lambda c: rank_value(c.rank))
