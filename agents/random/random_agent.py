import random
from collections import Counter

from environments.wist.actions import BidAction, PassAction, PlayCardAction
from environments.wist.observation import BiddingObservation, WistObservation
from environments.wist.rules import legal_cards
from intelligence.core.action import Action
from intelligence.core.agent import Agent
from intelligence.core.observation import Observation


class RandomAgent(Agent):
    """
    A simple Wist agent that randomly chooses from legal cards
    and makes random (but valid) bidding decisions.
    """

    def act(self, observation: Observation) -> Action:
        if isinstance(observation, BiddingObservation):
            return self._act_bidding(observation)

        if isinstance(observation, WistObservation):
            return self._act_play(observation)

        raise TypeError(
            f"RandomAgent does not support {type(observation).__name__}."
        )

    def _act_play(self, observation: WistObservation) -> Action:
        if not observation.hand:
            raise ValueError("RandomAgent cannot act with an empty hand.")

        leading_suit = None

        if observation.current_trick is not None:
            leading_suit = observation.current_trick.leading_suit

        must_lead_trump = None
        if observation.must_lead_trump and observation.trump_suit is not None:
            must_lead_trump = observation.trump_suit

        card = random.choice(
            legal_cards(
                hand=observation.hand,
                leading_suit=leading_suit,
                must_lead_trump=must_lead_trump,
            )
        )

        return PlayCardAction(
            player_id=observation.player_id,
            card=card,
        )

    def _act_bidding(self, observation: BiddingObservation) -> Action:
        """
        Random bidding strategy:
        - 50% chance to pass if not Sahib Al-Qabool.
        - Otherwise bid randomly within valid range.
        - Sahib Al-Qabool: 50% chance to accept (pass), 50% to match/outbid.
        """

        hand = observation.hand
        suit_counts = Counter(card.suit for card in hand)
        longest_suit_count = max(suit_counts.values()) if suit_counts else 0

        # Cannot bid with 8+ in one suit (should have been Dak).
        if longest_suit_count >= 8:
            return PassAction(player_id=observation.player_id)

        max_bid = min(longest_suit_count + 3, 13)

        if observation.is_sahib_al_qabool:
            # Sahib Al-Qabool: 50% accept the current bid, 50% match/outbid.
            if observation.current_highest_bid is not None and random.random() < 0.5:
                # Accept: pass.
                return PassAction(player_id=observation.player_id)

            # Match or outbid.
            min_bid = observation.current_highest_bid or 7

            if observation.is_opening_bid:
                min_bid = 7
                max_bid = min(max_bid, 11)

            if min_bid > max_bid:
                return PassAction(player_id=observation.player_id)

            bid_value = random.randint(min_bid, max_bid)

            return BidAction(
                player_id=observation.player_id,
                value=bid_value,
            )

        else:
            # Regular player: 50% pass, 50% bid.
            if random.random() < 0.5:
                return PassAction(player_id=observation.player_id)

            if observation.is_opening_bid:
                min_bid = 7
                max_bid_allowed = min(max_bid, 11)
            else:
                min_bid = (observation.current_highest_bid or 6) + 1
                max_bid_allowed = max_bid

            if min_bid > max_bid_allowed:
                return PassAction(player_id=observation.player_id)

            bid_value = random.randint(min_bid, max_bid_allowed)

            return BidAction(
                player_id=observation.player_id,
                value=bid_value,
            )