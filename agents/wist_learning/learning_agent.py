"""
Learning Wist Agent — TIBRAIN-backed Architecture.

Uses TIBRAIN's generic Agent with Double Q-learning, TD(λ), and prioritized
experience replay. The Wist layer provides domain-specific encoding and
game-logic while delegating all RL operations to TIBRAIN components.

The agent maintains:
- A TIBRAIN Agent for card-play decisions (Double Q + TD(λ) + replay)
- A separate bid Q-table (Monte Carlo first-visit updates)
- card_memory: tracks played cards within a shota
- opponent_model: learned opponent tendencies

External API is identical to the previous inline implementation.
"""

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from tibrain.agent import Agent as TIBRAINAgent
from tibrain.q_learning import QLearningEngine
from tibrain.policy import Policy
from tibrain.replay_buffer import ReplayBuffer

from environments.wist.actions import BidAction, PassAction, PlayCardAction
from environments.wist.observation import BiddingObservation, WistObservation
from environments.wist.rules import legal_cards, rank_value
from intelligence.core.action import Action
from intelligence.core.agent import Agent
from intelligence.core.cards.card import Card
from intelligence.core.cards.rank import Rank
from intelligence.core.cards.suit import Suit
from intelligence.core.observation import Observation


# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------

SUIT_INDEX = {Suit.SPADES: 0, Suit.HEARTS: 1, Suit.CLUBS: 2, Suit.DIAMONDS: 3}

HIGH_RANKS = {Rank.ACE, Rank.KING, Rank.QUEEN, Rank.JACK}

RANK_ORDER = [
    Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX,
    Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.TEN,
    Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE
]


# ---------------------------------------------------------------
# Opponent Model — tracks opponent tendencies
# ---------------------------------------------------------------


class OpponentModel:
    """
    Lightweight opponent model that tracks observable patterns:
    - Which suits opponents have shown void in (didn't follow)
    - How often opponents trump when void
    - High card usage patterns per game phase
    """

    def __init__(self):
        # opponent_id → set of suits they've shown void in
        self.known_voids: dict[int, set] = defaultdict(set)
        # opponent_id → count of times they trumped when void
        self.trump_tendency: dict[int, int] = defaultdict(int)
        self.void_opportunities: dict[int, int] = defaultdict(int)

    def observe_play(self, player_id: int, card: Card,
                     leading_suit: Suit | None, trump: Suit | None):
        """Record what we learn from an opponent's play."""
        if leading_suit and card.suit != leading_suit:
            # They didn't follow suit — they're void.
            self.known_voids[player_id].add(leading_suit)
            self.void_opportunities[player_id] += 1
            if trump and card.suit == trump:
                self.trump_tendency[player_id] += 1

    def is_void(self, player_id: int, suit: Suit) -> bool:
        """Check if we know a player is void in a suit."""
        return suit in self.known_voids[player_id]

    def trump_aggression(self, player_id: int) -> float:
        """How often does this opponent trump when they can? 0.0 - 1.0"""
        opps = self.void_opportunities[player_id]
        if opps == 0:
            return 0.5  # Unknown — assume moderate
        return self.trump_tendency[player_id] / opps

    def reset(self):
        self.known_voids.clear()
        self.trump_tendency.clear()
        self.void_opportunities.clear()

    def encode_context(self, my_player_id: int, trump: Suit | None) -> str:
        """
        Encode opponent model state into a compact string feature.
        Returns: opponent void pressure level (how many known voids for opponents).
        """
        my_team = {my_player_id, (my_player_id + 2) % 4}
        opp_ids = [pid for pid in range(4) if pid not in my_team]
        total_voids = sum(len(self.known_voids[pid]) for pid in opp_ids)
        if total_voids == 0:
            return "0"
        elif total_voids <= 2:
            return "1"
        else:
            return "2"


# ---------------------------------------------------------------
# State Encoding — Rich, multi-dimensional
# ---------------------------------------------------------------


def _count_high_remaining(suit: Suit, rank: Rank, played_cards: set) -> int:
    """Count how many cards higher than `rank` in `suit` are still unplayed."""
    rv = rank_value(rank)
    count = 0
    for r in RANK_ORDER:
        if rank_value(r) > rv:
            card_key = (suit, r)
            if card_key not in played_cards:
                count += 1
    return count


def encode_play_state(obs: WistObservation, card_memory: set,
                     opponent_model: OpponentModel | None = None) -> str:
    """
    Enhanced state encoding for card play decisions.

    Features (15 dimensions):
    - Position in trick: 0/1/2/3
    - Trump strength: 0/1/2/3 (none / 1-2 / 3-4 / 5+)
    - High cards in hand: 0/1/2/3 (A,K,Q,J count buckets)
    - Can follow suit: Y/N
    - Partner winning: Y/N/U (yes/no/unknown)
    - Game phase: E(arly 10+) / M(id 6-9) / L(ate 1-5)
    - Trick number bucket: A(1-4) / B(5-9) / C(10-13)
    - Team tricks differential: W(+3) / A(+1-2) / T(ied) / B(ehind) / F(ar behind -3)
    - Leading suit is trump: Y/N
    - Trump dominance: D(ominant) / S(trong) / W(eak) / N(one)
    - Void count: 0/1/2/3
    - Cards seen ratio: L(ow <25%) / M(id 25-60%) / H(igh >60%)
    - Seek potential: S(eeking - won all so far) / P(ossible) / N(o)
    - Opponent void pressure: 0/1/2
    """
    hand = obs.hand
    trump = obs.trump_suit

    # Position in trick.
    n_played = 0
    leading_suit = None
    if obs.current_trick and obs.current_trick.played_cards:
        n_played = len(obs.current_trick.played_cards)
        leading_suit = obs.current_trick.leading_suit
    pos = str(n_played)

    # Trump strength (finer grained).
    trump_count = sum(1 for c in hand if c.suit == trump) if trump else 0
    if trump_count == 0:
        ts = "0"
    elif trump_count <= 2:
        ts = "1"
    elif trump_count <= 4:
        ts = "2"
    else:
        ts = "3"

    # High card count (A, K, Q, J).
    highs = sum(1 for c in hand if c.rank in HIGH_RANKS)
    hc = str(min(highs, 3))

    # Can follow suit.
    cf = "Y" if (leading_suit and any(c.suit == leading_suit for c in hand)) else "N"

    # Partner winning current trick.
    pw = "U"
    if obs.current_trick and obs.current_trick.played_cards and trump:
        best_id = obs.current_trick.played_cards[0].player_id
        best_card = obs.current_trick.played_cards[0].card
        ls = obs.current_trick.leading_suit
        for pc in obs.current_trick.played_cards[1:]:
            c = pc.card
            if c.suit == trump and best_card.suit != trump:
                best_id, best_card = pc.player_id, c
            elif c.suit == trump and best_card.suit == trump:
                if rank_value(c.rank) > rank_value(best_card.rank):
                    best_id, best_card = pc.player_id, c
            elif c.suit == ls and best_card.suit == ls and best_card.suit != trump:
                if rank_value(c.rank) > rank_value(best_card.rank):
                    best_id, best_card = pc.player_id, c
        my_team = {obs.player_id, (obs.player_id + 2) % 4}
        pw = "Y" if best_id in my_team else "N"

    # Game phase (cards left in hand).
    cl = len(hand)
    ph = "E" if cl >= 10 else ("M" if cl >= 6 else "L")

    # Trick number bucket.
    trick_num = 14 - cl
    tn = "A" if trick_num <= 4 else ("B" if trick_num <= 9 else "C")

    # Team tricks differential.
    scores = obs.team_scores
    my_team_id = 0 if obs.player_id in (0, 2) else 1
    opp_team_id = 1 - my_team_id
    my_tricks = scores.get(my_team_id, 0)
    opp_tricks = scores.get(opp_team_id, 0)
    diff = my_tricks - opp_tricks
    if diff >= 3:
        td = "W"
    elif diff >= 1:
        td = "A"
    elif diff == 0:
        td = "T"
    elif diff >= -2:
        td = "B"
    else:
        td = "F"

    # Leading suit is trump.
    lt = "Y" if (leading_suit and leading_suit == trump) else "N"

    # Trump dominance — do I hold the highest remaining trump?
    td_dom = "N"
    if trump and trump_count > 0:
        my_trumps = [c for c in hand if c.suit == trump]
        my_best_trump = max(my_trumps, key=lambda c: rank_value(c.rank))
        higher_unplayed = _count_high_remaining(trump, my_best_trump.rank, card_memory)
        if higher_unplayed == 0:
            td_dom = "D"  # Dominant — I hold top trump
        elif higher_unplayed <= 1:
            td_dom = "S"  # Strong
        else:
            td_dom = "W"  # Weak

    # Void count.
    suits_in_hand = set(c.suit for c in hand)
    voids = 4 - len(suits_in_hand)
    vc = str(min(voids, 3))

    # Cards seen ratio.
    seen_pct = len(card_memory) / 52.0
    cs = "L" if seen_pct < 0.25 else ("M" if seen_pct < 0.60 else "H")

    # Seek potential — are we on track for seek?
    total_tricks = my_tricks + opp_tricks
    if total_tricks > 0 and opp_tricks == 0:
        sp = "S"  # Actively seeking — won all tricks so far
    elif total_tricks > 0 and opp_tricks <= 1 and my_tricks >= 4:
        sp = "P"  # Possible — opponent barely has tricks
    else:
        sp = "N"

    # Opponent void pressure from opponent model.
    ovp = "0"
    if opponent_model:
        ovp = opponent_model.encode_context(obs.player_id, trump)

    return f"{pos}{ts}{hc}{cf}{pw}{ph}{tn}{td}{lt}{td_dom}{vc}{cs}{sp}{ovp}"


def encode_play_action(card: Card, trump: Suit | None, leading_suit: Suit | None,
                       card_memory: set) -> str:
    """
    Enhanced action encoding for card play.

    Encodes the CONTEXT of the play (4 features):
    - Is it trump? T/N
    - Is it following suit? F/O (follow/off-suit)
    - Rank tier: H(igh: A,K), U(pper-mid: Q,J), M(id: 10,9,8), L(ow: 7-2)
    - Is it the highest remaining in its suit? Y/N
    """
    is_trump = "T" if card.suit == trump else "N"
    follows = "F" if (leading_suit and card.suit == leading_suit) else "O"

    rv = rank_value(card.rank)
    if rv >= 13:  # K, A
        tier = "H"
    elif rv >= 11:  # Q, J
        tier = "U"
    elif rv >= 8:  # 10, 9, 8
        tier = "M"
    else:
        tier = "L"

    # Is this the highest remaining card in its suit?
    higher_unplayed = _count_high_remaining(card.suit, card.rank, card_memory)
    top = "Y" if higher_unplayed == 0 else "N"

    return f"{is_trump}{follows}{tier}{top}"


def encode_bid_state(obs: BiddingObservation) -> str:
    """
    Enhanced state encoding for bidding decisions.

    Features (7 dimensions):
    - Longest suit count: 4/5/6/7
    - High cards in longest suit: 0/1/2/3+
    - Total high cards: 0/1/2/3/4+
    - Void suits: 0/1/2
    - Has existing bid: Y/N
    - Is Qabool: Y/N
    - Side suit strength: W(eak 0-1 high) / M(od 2) / S(trong 3+)
    """
    hand = obs.hand
    suit_counts = Counter(card.suit for card in hand)
    longest = max(suit_counts.values()) if suit_counts else 0
    longest_suit = max(suit_counts, key=suit_counts.get) if suit_counts else None

    # Longest suit bucket.
    ls = str(min(longest, 7))

    # High cards in longest suit.
    highs_in_longest = sum(1 for c in hand
                           if c.suit == longest_suit and c.rank in HIGH_RANKS)
    hl = str(min(highs_in_longest, 3))

    # Total high cards.
    total_highs = sum(1 for c in hand if c.rank in HIGH_RANKS)
    th = str(min(total_highs, 4))

    # Void suits.
    suits_present = set(c.suit for c in hand)
    voids = 4 - len(suits_present)
    vs = str(min(voids, 2))

    # Has existing bid.
    hb = "Y" if obs.current_highest_bid else "N"

    # Is Qabool.
    iq = "Y" if obs.is_sahib_al_qabool else "N"

    # Side suit strength.
    side_highs = sum(1 for c in hand
                     if c.suit != longest_suit and c.rank in HIGH_RANKS)
    ss = "W" if side_highs <= 1 else ("M" if side_highs == 2 else "S")

    return f"{ls}{hl}{th}{vs}{hb}{iq}{ss}"


def encode_bid_action(action: Action) -> str:
    """Encode bidding action into finer categories."""
    if isinstance(action, PassAction):
        return "PASS"
    if isinstance(action, BidAction):
        v = action.value
        if v <= 7:
            return "B7"
        elif v <= 8:
            return "B8"
        elif v <= 9:
            return "B9"
        elif v <= 10:
            return "B10"
        elif v <= 11:
            return "B11"
        else:
            return "B12+"
    return "PASS"


# ---------------------------------------------------------------
# Wist-specific Encoders for TIBRAIN Agent
# ---------------------------------------------------------------


class WistPlayStateEncoder:
    """Encodes a (WistObservation, card_memory, opponent_model) tuple to a string key.

    The TIBRAIN Agent calls this with the state object provided to choose_action/learn.
    We pass tuples of (obs, card_memory, opponent_model) as the "state".
    """

    def __call__(self, state) -> str:
        obs, card_memory, opponent_model = state
        return encode_play_state(obs, card_memory, opponent_model)


class WistPlayActionEncoder:
    """Encodes a (Card, trump, leading_suit, card_memory) tuple to a string key.

    The TIBRAIN Agent calls this with the action object provided to choose_action/learn.
    We pass tuples of (card, trump, leading_suit, card_memory) as the "action".
    """

    def __call__(self, action) -> str:
        card, trump, leading_suit, card_memory = action
        return encode_play_action(card, trump, leading_suit, card_memory)


# ---------------------------------------------------------------
# Learning Agent — TIBRAIN-backed
# ---------------------------------------------------------------


class LearningAgent(Agent):
    """
    Wist learning agent backed by TIBRAIN's generic RL components.

    Delegates Q-learning, policy selection, and experience replay to a
    TIBRAIN Agent instance. Wist-specific logic (state encoding, action
    encoding, bidding, card memory, opponent modeling, reward shaping)
    remains in this layer.

    External API is identical to the previous inline implementation:
    - act(observation) -> Action
    - reward_trick(won: bool) -> None
    - reward_shota(...) -> None
    - reset_episode() -> None
    - observe_card_played(card) -> None
    - decay_epsilon(...) -> None
    - decay_alpha(...) -> None
    - save(path) -> None
    - load(path) -> LearningAgent
    - q_table_size -> int
    """

    def __init__(self, epsilon: float = 0.3, training: bool = True,
                 alpha: float = 0.1, gamma: float = 0.95,
                 lambda_: float = 0.7) -> None:
        # TIBRAIN Agent for card-play decisions.
        self._tibrain_agent = TIBRAINAgent(
            state_encoder=WistPlayStateEncoder(),
            action_encoder=WistPlayActionEncoder(),
            alpha=alpha,
            gamma=gamma,
            lambda_trace=lambda_,
            epsilon=epsilon,
            epsilon_min=0.01,
            training=training,
            replay_capacity=8000,
        )

        # Bid Q-table (kept separate — uses Monte Carlo first-visit updates).
        self.bid_q: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.bid_n: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Hyperparameters (exposed for external decay and save/load).
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        self.training = training

        # Episode memory.
        self._bid_episode: list[tuple[str, str]] = []
        self._last_state = None
        self._last_action = None

        # Card memory — tracks played cards within a shota.
        self._card_memory: set[tuple[Suit, Rank]] = set()
        self._tricks_won_this_shota: int = 0
        self._tricks_lost_this_shota: int = 0

        # Opponent model.
        self._opponent_model = OpponentModel()

        self.total_updates = 0
        self.episodes_trained = 0

    @property
    def _q_engine(self) -> QLearningEngine:
        """Access to the underlying Q-learning engine (for save/load)."""
        return self._tibrain_agent.q_engine

    @property
    def _policy(self) -> Policy:
        """Access to the underlying policy (for epsilon sync)."""
        return self._tibrain_agent.policy

    def act(self, observation: Observation) -> Action:
        if isinstance(observation, BiddingObservation):
            return self._act_bidding(observation)
        if isinstance(observation, WistObservation):
            return self._act_play(observation)
        raise TypeError(f"Unsupported observation: {type(observation).__name__}")

    # ----------------------------------------------------------
    # Card play with TIBRAIN Agent delegation
    # ----------------------------------------------------------

    def _act_play(self, obs: WistObservation) -> Action:
        if not obs.hand:
            raise ValueError("Cannot act with empty hand.")

        leading_suit = None
        if obs.current_trick:
            leading_suit = obs.current_trick.leading_suit

        must_lead_trump = None
        if obs.must_lead_trump and obs.trump_suit:
            must_lead_trump = obs.trump_suit

        playable = legal_cards(obs.hand, leading_suit, must_lead_trump)

        if len(playable) == 1:
            card = playable[0]
        else:
            # Build state and legal actions in TIBRAIN's expected format.
            state = (obs, self._card_memory, self._opponent_model)
            legal_actions = [
                (c, obs.trump_suit, leading_suit, self._card_memory)
                for c in playable
            ]

            # Delegate action selection to TIBRAIN Agent.
            chosen_action = self._tibrain_agent.choose_action(state, legal_actions)
            card = chosen_action[0]  # Extract the Card from the tuple

        # Record state-action for TD learning.
        if self.training:
            state_tuple = (obs, self._card_memory, self._opponent_model)
            action_tuple = (card, obs.trump_suit, leading_suit, self._card_memory)

            # TD update from previous step.
            if self._last_state is not None:
                # Learn from the transition: prev_state → 0 reward → current state
                self._tibrain_agent.learn(
                    self._last_state,
                    self._last_action,
                    0.0,
                    state_tuple,
                    [(c, obs.trump_suit, leading_suit, self._card_memory) for c in playable],
                )
                self.total_updates += 1

            self._last_state = state_tuple
            self._last_action = action_tuple

        # Update card memory and opponent model from current trick.
        if obs.current_trick:
            for pc in obs.current_trick.played_cards:
                self._card_memory.add((pc.card.suit, pc.card.rank))
                # Opponent modeling: record what opponents played.
                if pc.player_id != obs.player_id:
                    self._opponent_model.observe_play(
                        pc.player_id, pc.card, leading_suit, obs.trump_suit)

        return PlayCardAction(player_id=obs.player_id, card=card)

    # ----------------------------------------------------------
    # Bidding (learned — kept in Wist layer with Monte Carlo)
    # ----------------------------------------------------------

    def _act_bidding(self, obs: BiddingObservation) -> Action:
        hand = obs.hand
        suit_counts = Counter(card.suit for card in hand)
        longest = max(suit_counts.values()) if suit_counts else 0

        # Can't bid with 8+ in one suit (Dak territory).
        if longest >= 8:
            action = PassAction(player_id=obs.player_id)
            if self.training:
                state = encode_bid_state(obs)
                self._bid_episode.append((state, encode_bid_action(action)))
            return action

        # Calculate feasible bid range.
        min_bid = longest + 3
        min_bid = max(7, min_bid)

        # Explore or exploit.
        if self.training and random.random() < self.epsilon:
            action = self._random_bid(obs, min_bid, longest)
        else:
            action = self._best_bid(obs, min_bid, longest)

        if self.training:
            state = encode_bid_state(obs)
            self._bid_episode.append((state, encode_bid_action(action)))

        return action

    def _random_bid(self, obs: BiddingObservation, min_bid: int, longest: int) -> Action:
        """Random exploration bid."""
        # 35% chance to pass, 65% chance to bid.
        if random.random() < 0.35:
            return PassAction(player_id=obs.player_id)

        bid_value = random.randint(max(7, min_bid), min(13, min_bid + 3))

        # Respect constraints.
        if obs.current_highest_bid and bid_value <= obs.current_highest_bid:
            if obs.is_sahib_al_qabool:
                bid_value = obs.current_highest_bid  # Qabool can match.
            else:
                return PassAction(player_id=obs.player_id)

        if not obs.is_sahib_al_qabool and obs.is_opening_bid and bid_value > 11:
            bid_value = 11

        return BidAction(player_id=obs.player_id, value=bid_value)

    def _best_bid(self, obs: BiddingObservation, min_bid: int, longest: int) -> Action:
        """Choose best bid based on learned Q-values."""
        state = encode_bid_state(obs)
        q_values = self.bid_q[state]

        # Get Q-values for each possible action.
        actions = ["PASS", "B7", "B8", "B9", "B10", "B11", "B12+"]
        best_action = "PASS"
        best_q = q_values.get("PASS", 0.0)

        for key in actions[1:]:
            q = q_values.get(key, 0.0)
            if q > best_q:
                best_q = q
                best_action = key

        if best_action == "PASS":
            return PassAction(player_id=obs.player_id)

        # Convert to actual bid value.
        action_to_value = {
            "B7": 7, "B8": 8, "B9": 9, "B10": 10, "B11": 11, "B12+": 12
        }
        bid_value = action_to_value.get(best_action, max(7, min_bid))
        bid_value = max(bid_value, min_bid)
        bid_value = min(bid_value, 13)

        # Respect opening bid max 11 rule.
        if not obs.is_sahib_al_qabool and obs.is_opening_bid and bid_value > 11:
            bid_value = 11

        # Respect constraints.
        if obs.current_highest_bid:
            if obs.is_sahib_al_qabool:
                if bid_value < obs.current_highest_bid:
                    bid_value = obs.current_highest_bid
            else:
                if bid_value <= obs.current_highest_bid:
                    return PassAction(player_id=obs.player_id)

        return BidAction(player_id=obs.player_id, value=bid_value)

    # ----------------------------------------------------------
    # Reward signals — per-trick and end-of-shota
    # ----------------------------------------------------------

    def reward_trick(self, won: bool) -> None:
        """
        Per-trick reward signal with strategic shaping.

        Rewards consider:
        - Basic win/loss
        - Seek potential (winning all tricks = massive bonus building)
        - Consecutive wins (momentum bonus)
        """
        if not self.training:
            return

        if won:
            self._tricks_won_this_shota += 1
            # Base reward + seek momentum bonus.
            if self._tricks_lost_this_shota == 0:
                # On seek track — escalating reward.
                seek_bonus = 0.05 * self._tricks_won_this_shota
                trick_reward = 0.3 + seek_bonus
            else:
                trick_reward = 0.25
        else:
            self._tricks_lost_this_shota += 1
            if self._tricks_lost_this_shota == 1 and self._tricks_won_this_shota >= 5:
                # Lost seek after a long streak — harsh signal.
                trick_reward = -0.5
            else:
                trick_reward = -0.15

        # Apply trick reward via TIBRAIN learn (terminal-like step with reward).
        if self._last_state is not None:
            # Use a "terminal" learn step: next state is same as last (no meaningful next).
            # The reward is the trick shaping signal.
            self._tibrain_agent.learn(
                self._last_state,
                self._last_action,
                trick_reward,
                self._last_state,  # pseudo next-state
                [],  # no next actions (terminal-like for this trick)
            )
            self.total_updates += 1

    def reward_shota(self, team_won_shota: bool, bid_met: bool,
                     my_tricks: int = 0, opp_tricks: int = 0,
                     was_shooter: bool = False, seek: bool = False) -> None:
        """
        End-of-Shota reward with enhanced shaping.

        Reward structure:
        - Base: +1.5 for winning shota, -1.5 for losing
        - Bid bonus: +1.0 if our team met the bid as shooter
        - Bid penalty: -0.8 if we were shooter and failed
        - Trick margin: +0.15 per trick above opponent (diminishing)
        - Seek bonus: +3.0 for achieving seek
        - Seek penalty: -2.0 for being seeked against
        - Dominance bonus: +0.5 for winning 10+ tricks (near-seek)
        """
        if not self.training:
            self._reset_episode_state()
            return

        # Compute return.
        G = 1.5 if team_won_shota else -1.5

        # Bid bonus/penalty (stronger signal).
        if was_shooter:
            G += 1.0 if bid_met else -0.8

        # Trick margin (diminishing returns).
        margin = my_tricks - opp_tricks
        G += min(margin * 0.15, 1.5)

        # Seek signals.
        if seek:
            if my_tricks == 13:
                G += 3.0
            else:
                G -= 2.0

        # Near-seek bonus (domination without full seek).
        if my_tricks >= 10 and not seek:
            G += 0.5

        # Clamp to reasonable range.
        G = max(-4.0, min(5.0, G))

        # Final TD update — terminal state with shota reward.
        if self._last_state is not None:
            self._tibrain_agent.learn(
                self._last_state,
                self._last_action,
                G,
                self._last_state,  # terminal pseudo next-state
                [],  # no next actions (episode end)
            )
            self.total_updates += 1

        # Update bid Q-table with Monte Carlo (bids are one-shot).
        self._update_bid_table(G)

        self.episodes_trained += 1
        self._reset_episode_state()

    def _update_bid_table(self, G: float):
        """First-visit Monte Carlo update on bid Q-table."""
        visited = set()
        for state, action_key in self._bid_episode:
            sa = (state, action_key)
            if sa in visited:
                continue
            visited.add(sa)

            self.bid_n[state][action_key] += 1
            n = self.bid_n[state][action_key]
            old_q = self.bid_q[state][action_key]
            # Weighted average with recency bias.
            weight = min(1.0 / n, self.alpha)
            self.bid_q[state][action_key] = old_q + weight * (G - old_q)
            self.total_updates += 1

    def _reset_episode_state(self):
        """Clear all episode-specific state."""
        self._bid_episode.clear()
        self._last_state = None
        self._last_action = None
        self._card_memory.clear()
        self._tricks_won_this_shota = 0
        self._tricks_lost_this_shota = 0
        self._opponent_model.reset()
        # Reset TIBRAIN agent episode (clears eligibility traces).
        self._tibrain_agent.reset_episode()

    def reset_episode(self) -> None:
        """Clear episode memory (call on Dak / skipped Shota)."""
        self._reset_episode_state()

    def observe_card_played(self, card: Card) -> None:
        """
        Called externally when any card is played (by any player).
        Maintains card memory for state encoding.
        """
        self._card_memory.add((card.suit, card.rank))

    def decay_epsilon(self, min_epsilon: float = 0.03, decay_rate: float = 0.9997) -> None:
        """Slowly reduce exploration. Decays slower than before for better coverage."""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
        # Sync epsilon with TIBRAIN policy.
        self._tibrain_agent.policy.epsilon = self.epsilon

    def decay_alpha(self, min_alpha: float = 0.01, decay_rate: float = 0.9999) -> None:
        """Decay learning rate for convergence."""
        self.alpha = max(min_alpha, self.alpha * decay_rate)
        # Sync alpha with TIBRAIN Q-learning engine.
        self._tibrain_agent.q_engine.alpha = self.alpha

    # ----------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Extract Q-table data from TIBRAIN engine.
        q1_data = self._q_engine.q1.to_dict()
        q2_data = self._q_engine.q2.to_dict()

        data = {
            "version": 4,  # Schema version for TIBRAIN-backed agent.
            "q_table": q1_data,
            "q_table2": q2_data,
            "bid_q": {k: dict(v) for k, v in self.bid_q.items()},
            "bid_n": {k: dict(v) for k, v in self.bid_n.items()},
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "lambda": self.lambda_,
            "total_updates": self.total_updates,
            "episodes_trained": self.episodes_trained,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str | Path, training: bool = False) -> "LearningAgent":
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)

        agent = cls(
            epsilon=data.get("epsilon", 0.05),
            training=training,
            alpha=data.get("alpha", 0.1),
            gamma=data.get("gamma", 0.95),
            lambda_=data.get("lambda", 0.7),
        )

        # Load Q-tables into TIBRAIN engine.
        from tibrain.q_table import QTable
        agent._q_engine.q1 = QTable.from_dict(data.get("q_table", {}))
        agent._q_engine.q2 = QTable.from_dict(data.get("q_table2", {}))

        # Load bid Q-table.
        for state, actions in data.get("bid_q", {}).items():
            for action_key, value in actions.items():
                agent.bid_q[state][action_key] = value
        for state, actions in data.get("bid_n", {}).items():
            for action_key, value in actions.items():
                agent.bid_n[state][action_key] = value

        agent.total_updates = data.get("total_updates", 0)
        agent.episodes_trained = data.get("episodes_trained", 0)
        return agent

    @property
    def q_table(self) -> dict[str, dict[str, float]]:
        """Backward-compatible access to play Q-values (averaged Q1 + Q2).

        Returns a dict-of-dicts view suitable for external code that reads
        Q-values by state key (e.g., GUI advisor panels).
        """
        q1_data = self._q_engine.q1.to_dict()
        q2_data = self._q_engine.q2.to_dict()
        merged: dict[str, dict[str, float]] = {}
        all_states = set(q1_data.keys()) | set(q2_data.keys())
        for state in all_states:
            actions_1 = q1_data.get(state, {})
            actions_2 = q2_data.get(state, {})
            all_actions = set(actions_1.keys()) | set(actions_2.keys())
            merged[state] = {
                a: (actions_1.get(a, 0.0) + actions_2.get(a, 0.0)) / 2.0
                for a in all_actions
            }
        return merged

    @property
    def q_table_size(self) -> int:
        play_size = self._q_engine.q1.size + self._q_engine.q2.size
        bid_size = sum(len(v) for v in self.bid_q.values())
        return play_size + bid_size
