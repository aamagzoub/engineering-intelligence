"""
Wist Discovery Agent — learns Wist strategy from scratch.

This agent has ZERO domain knowledge about Wist. It does not know:
- That trump beats other suits
- How bidding works strategically
- That seek (13 tricks) wins instantly
- What makes a good bid value
- Any strategy whatsoever

It receives ONLY:
- Observation: hand + legal moves + visible trick state
- Reward: numeric score at the end of each shota

Architecture:
- Q-learning with Monte Carlo credit assignment
- Separate Q-tables for bidding and trick play
- Domain-agnostic state encoding (hand shape, position, relative strengths)
- Epsilon-greedy exploration with decay
- Self-play: opponents share Q-tables
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from environments.wist.actions import BidAction, PassAction, PlayCardAction
from environments.wist.observation import BiddingObservation, WistObservation
from environments.wist.rules import legal_cards, rank_value
from intelligence.core.action import Action
from intelligence.core.agent import Agent
from intelligence.core.cards.rank import Rank
from intelligence.core.cards.suit import Suit
from intelligence.core.observation import Observation


# Domain-agnostic rank ordering (agent doesn't know names, just relative values).
RANK_VAL = {
    Rank.TWO: 2, Rank.THREE: 3, Rank.FOUR: 4, Rank.FIVE: 5, Rank.SIX: 6,
    Rank.SEVEN: 7, Rank.EIGHT: 8, Rank.NINE: 9, Rank.TEN: 10,
    Rank.JACK: 11, Rank.QUEEN: 12, Rank.KING: 13, Rank.ACE: 14,
}
SUIT_IDX = {Suit.SPADES: 0, Suit.HEARTS: 1, Suit.CLUBS: 2, Suit.DIAMONDS: 3}


def _encode_play_state(obs: WistObservation) -> str:
    """Rich state encoding — observable features only, no strategy knowledge."""
    hand = obs.hand
    n_cards = len(hand)

    # Suit distribution (sorted descending).
    suits = [0, 0, 0, 0]
    highs = 0  # Cards rank >= Q
    aces = 0
    trump_count = 0
    trump_highs = 0
    for c in hand:
        suits[SUIT_IDX[c.suit]] += 1
        if RANK_VAL[c.rank] >= 12:
            highs += 1
        if RANK_VAL[c.rank] == 14:
            aces += 1
        if obs.trump_suit and c.suit == obs.trump_suit:
            trump_count += 1
            if RANK_VAL[c.rank] >= 12:
                trump_highs += 1

    shape = "".join(str(min(s, 9)) for s in sorted(suits, reverse=True))

    # Trick position (0-3 cards played before us).
    pos = 0
    if obs.current_trick and obs.current_trick.played_cards:
        pos = len(obs.current_trick.played_cards)

    # Game phase (more granular).
    if n_cards >= 11:
        phase = "1"  # Opening (tricks 1-2)
    elif n_cards >= 8:
        phase = "2"  # Early (tricks 3-5)
    elif n_cards >= 5:
        phase = "3"  # Mid (tricks 6-8)
    elif n_cards >= 2:
        phase = "4"  # Late (tricks 9-11)
    else:
        phase = "5"  # Endgame (tricks 12-13)

    # Team score difference.
    my_team = 0 if obs.player_id in (0, 2) else 1
    opp_team = 1 - my_team
    my_t = obs.team_scores.get(my_team, 0)
    opp_t = obs.team_scores.get(opp_team, 0)
    diff = my_t - opp_t
    td = "W" if diff >= 3 else ("A" if diff > 0 else ("T" if diff == 0 else "B"))

    # Trump strength: how many trumps + are they high.
    ts = f"{min(trump_count, 7)}{min(trump_highs, 4)}"

    # Voids: how many suits we have 0 cards in.
    voids = sum(1 for s in suits if s == 0)

    return f"{shape}{pos}{phase}{td}{min(highs, 5)}{ts}v{voids}a{min(aces, 4)}"


def _encode_play_action(card, obs: WistObservation) -> str:
    """Richer action encoding — relative strength + trump awareness."""
    rv = RANK_VAL[card.rank]
    # Finer rank tiers.
    if rv == 14:
        tier = "A"  # Ace
    elif rv >= 12:
        tier = "H"  # High (K, Q)
    elif rv >= 9:
        tier = "M"  # Mid (9, 10, J)
    else:
        tier = "L"  # Low (2-8)

    # Is it following suit?
    leading = None
    if obs.current_trick and obs.current_trick.leading_suit:
        leading = obs.current_trick.leading_suit
    follows = "F" if (leading and card.suit == leading) else "O"

    # Is it trump?
    is_trump = "T" if (obs.trump_suit and card.suit == obs.trump_suit) else "N"

    # Is it from longest suit in hand?
    from collections import Counter
    suit_counts = Counter(c.suit for c in obs.hand)
    longest = max(suit_counts.values()) if suit_counts else 0
    is_long = "L" if suit_counts.get(card.suit, 0) == longest else "S"

    # Would this create a void? (last card of this suit)
    creates_void = "V" if suit_counts.get(card.suit, 0) == 1 else "K"

    return f"{tier}{follows}{is_trump}{is_long}{creates_void}"


def _encode_bid_state(obs: BiddingObservation) -> str:
    """Richer bid state — hand composition + context."""
    hand = obs.hand
    from collections import Counter
    suit_counts = Counter(c.suit for c in hand)
    longest = max(suit_counts.values()) if suit_counts else 0
    shortest_valid = min((c for c in suit_counts.values() if 1 <= c <= 7), default=0)
    highs = sum(1 for c in hand if RANK_VAL[c.rank] >= 12)
    aces = sum(1 for c in hand if RANK_VAL[c.rank] == 14)
    voids = sum(1 for s in [Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS]
                if suit_counts.get(s, 0) == 0)
    has_bid = "Y" if obs.current_highest_bid else "N"
    is_q = "Y" if obs.is_sahib_al_qabool else "N"
    forced = "F" if obs.must_play else "N"
    bid_level = str(min(obs.current_highest_bid, 13)) if obs.current_highest_bid else "0"

    return f"{longest}{shortest_valid}{min(highs, 5)}{min(aces, 4)}v{voids}{has_bid}{bid_level}{is_q}{forced}"


def _encode_bid_action(action: Action) -> str:
    """Encode bid action."""
    if isinstance(action, PassAction):
        return "PASS"
    if isinstance(action, BidAction):
        return f"B{min(action.value, 13)}"
    return "PASS"


class WistDiscoveryAgent(Agent):
    """
    Wist Discovery Agent — learns entirely from reward signals.
    No domain knowledge. No hard-coded strategy.
    """

    def __init__(self, epsilon: float = 0.4, alpha: float = 0.2,
                 gamma: float = 0.97, training: bool = True) -> None:
        self.play_q: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.bid_q: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.training = training

        self._play_episode: list[tuple[str, str]] = []
        self._bid_episode: list[tuple[str, str]] = []

        self.episodes_trained: int = 0
        self.total_updates: int = 0

    def act(self, observation: Observation) -> Action:
        if isinstance(observation, BiddingObservation):
            return self._act_bid(observation)
        if isinstance(observation, WistObservation):
            return self._act_play(observation)
        raise TypeError(f"Unsupported: {type(observation).__name__}")

    def _act_bid(self, obs: BiddingObservation) -> Action:
        """Bid or pass — learned from reward only."""
        from collections import Counter
        hand = obs.hand
        suit_counts = Counter(c.suit for c in hand)

        # Valid trump suits: 1–7 cards. (8+ suit exists but can't be used as trump.)
        valid_trump_suits = [s for s, count in suit_counts.items() if 1 <= count <= 7]

        if not valid_trump_suits:
            # Every suit has 8+ cards (extremely rare with 13 cards, but guard).
            return PassAction(player_id=obs.player_id)

        # Min bid depends on shortest valid trump suit (gives lowest floor).
        shortest_trump_count = min(suit_counts[s] for s in valid_trump_suits)

        # Qabool rules depend on whether someone else already bid.
        if obs.is_sahib_al_qabool:
            if obs.current_highest_bid:
                # Someone bid — Qabool exempt from trump+3, can match.
                min_bid = obs.current_highest_bid  # Can match (not exceed).
                max_bid = 13
            else:
                # All passed — Qabool bids first. Trump+3 applies, but cap is 13.
                min_bid = max(7, shortest_trump_count + 3)
                max_bid = 13
        else:
            # Regular player.
            min_bid = max(7, shortest_trump_count + 3)
            max_bid = 11 if obs.is_opening_bid else 13
            if obs.current_highest_bid:
                min_bid = max(min_bid, obs.current_highest_bid + 1)  # Must exceed.

        if min_bid > max_bid and not obs.must_play:
            action = PassAction(player_id=obs.player_id)
        elif min_bid > max_bid and obs.must_play:
            # Forced to play but no valid range — bid minimum possible.
            action = BidAction(player_id=obs.player_id, value=min(min_bid, 13))
        elif obs.must_play:
            # 3rd Dak: must bid, cannot pass.
            if self.training and random.random() < self.epsilon:
                bid_val = random.randint(min_bid, min(max_bid, min_bid + 2))
            else:
                best = self._best_bid(obs, min_bid, max_bid)
                if isinstance(best, BidAction):
                    bid_val = best.value
                else:
                    bid_val = min_bid
            action = BidAction(player_id=obs.player_id, value=bid_val)
        elif self.training and random.random() < self.epsilon:
            # Explore: 50% pass, 50% bid random valid value.
            if random.random() < 0.5:
                action = PassAction(player_id=obs.player_id)
            else:
                bid_val = random.randint(min_bid, min(max_bid, min_bid + 2))
                action = BidAction(player_id=obs.player_id, value=bid_val)
        else:
            action = self._best_bid(obs, min_bid, max_bid)

        if self.training:
            state = _encode_bid_state(obs)
            self._bid_episode.append((state, _encode_bid_action(action)))

        return action

    def _best_bid(self, obs: BiddingObservation, min_bid: int, max_bid: int = 13) -> Action:
        """Pick best bid from Q-table."""
        state = _encode_bid_state(obs)
        q = self.bid_q[state]

        best_action = "PASS"
        best_q = q.get("PASS", 0.0)

        for v in range(min_bid, max_bid + 1):
            key = f"B{v}"
            val = q.get(key, 0.0)
            if val > best_q:
                best_q = val
                best_action = key

        if best_action == "PASS":
            return PassAction(player_id=obs.player_id)

        bid_val = int(best_action[1:])
        return BidAction(player_id=obs.player_id, value=bid_val)

    def _act_play(self, obs: WistObservation) -> Action:
        """Play a card — learned from reward only."""
        leading_suit = None
        if obs.current_trick:
            leading_suit = obs.current_trick.leading_suit
        must_trump = obs.trump_suit if obs.must_lead_trump else None
        playable = legal_cards(obs.hand, leading_suit, must_trump)

        if len(playable) == 1:
            card = playable[0]
        elif self.training and random.random() < self.epsilon:
            card = random.choice(playable)
        else:
            card = self._best_card(obs, playable)

        if self.training:
            state = _encode_play_state(obs)
            action_key = _encode_play_action(card, obs)
            self._play_episode.append((state, action_key))

        return PlayCardAction(player_id=obs.player_id, card=card)

    def _best_card(self, obs: WistObservation, playable: list) -> object:
        """Pick card with best Q-value."""
        state = _encode_play_state(obs)
        q = self.play_q[state]
        best_card = playable[0]
        best_q = float("-inf")
        for card in playable:
            key = _encode_play_action(card, obs)
            val = q.get(key, 0.0)
            if val > best_q:
                best_q = val
                best_card = card
        return best_card

    def reward(self, score: float) -> None:
        """End-of-shota reward — the ONLY learning signal."""
        if not self.training:
            return

        # Adaptive learning rate — learn fast early, stabilize later.
        effective_alpha = max(0.05, self.alpha * (1.0 / (1.0 + self.episodes_trained / 1000)))

        # Update play Q-table (later actions get more credit).
        reward = score
        for state, action in reversed(self._play_episode):
            current_q = self.play_q[state][action]
            self.play_q[state][action] += effective_alpha * (reward - current_q)
            reward *= self.gamma
            self.total_updates += 1

        # Update bid Q-table.
        bid_reward = score * (self.gamma ** len(self._play_episode))
        for state, action in reversed(self._bid_episode):
            current_q = self.bid_q[state][action]
            self.bid_q[state][action] += effective_alpha * (bid_reward - current_q)
            self.total_updates += 1

        self._play_episode.clear()
        self._bid_episode.clear()
        self.episodes_trained += 1

    def reset_episode(self):
        """Clear episode memory (on Dak/skip)."""
        self._play_episode.clear()
        self._bid_episode.clear()

    def save(self, path: str) -> None:
        """Save model to JSON."""
        data = {
            "play_q": {k: dict(v) for k, v in self.play_q.items()},
            "bid_q": {k: dict(v) for k, v in self.bid_q.items()},
            "episodes_trained": self.episodes_trained,
            "total_updates": self.total_updates,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "gamma": self.gamma,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        """Load model from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        self.play_q = defaultdict(lambda: defaultdict(float))
        self.bid_q = defaultdict(lambda: defaultdict(float))
        for s, actions in data.get("play_q", {}).items():
            for a, v in actions.items():
                self.play_q[s][a] = v
        for s, actions in data.get("bid_q", {}).items():
            for a, v in actions.items():
                self.bid_q[s][a] = v
        self.episodes_trained = data.get("episodes_trained", 0)
        self.total_updates = data.get("total_updates", 0)
        self.epsilon = data.get("epsilon", self.epsilon)
