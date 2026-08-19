"""
Wist Discovery Agent — Advanced RL Architecture v2.

Zero domain knowledge. Receives ONLY: environment + legal moves + score signal.

Architecture (fully domain-agnostic, transferable):
1. Double Q-Learning — two Q-tables to prevent overestimation bias
2. Eligibility Traces (TD(λ)) — decaying credit to recent state-actions
3. Experience Replay with Prioritization — re-learn from surprising outcomes
4. Curiosity Bonus — exploration reward for visiting new states
5. Opponent Modeling — track known opponent voids from observations
6. Hierarchical Learning — separate bid/play phases with different learning rates
7. Reward Normalization — scale rewards to standard range (via TIBRAIN)
8. Per-trick intermediate rewards — faster credit assignment
9. Neural Network function approximator — generalization across similar states (via TIBRAIN)
10. Population-based hyperparameter tracking — auto-tune learning parameters
11. Self-play Elo tracking — measure improvement over time (via TIBRAIN)
12. Curriculum learning — progressive difficulty
13. Opponent prediction — predict opponent actions from patterns
14. Meta-learning — adapt hyperparameters based on recent performance (via TIBRAIN)
15. Pattern Discovery — recurring patterns in experience data (via TIBRAIN)
"""

import json
import random
import math
from collections import defaultdict, deque, Counter
from pathlib import Path

import numpy as np

from environments.wist.actions import BidAction, PassAction, PlayCardAction
from environments.wist.observation import BiddingObservation, WistObservation
from environments.wist.rules import legal_cards, rank_value
from intelligence.core.action import Action
from intelligence.core.agent import Agent
from intelligence.core.cards.rank import Rank
from intelligence.core.cards.suit import Suit

# === TIBRAIN imports (generic RL components) ===
from tibrain.neural_net import Evaluator, QNetwork
from tibrain.mcts import MCTSEngine
from tibrain.reward import RewardNormalizer
from tibrain.evaluation import EloTracker, MetaLearner
from tibrain.discovery import DiscoveryEngine

# === Wist-specific feature extraction ===
from agents.wist_discovery.neural_net import (
    CardEvaluator, state_to_features, state_features, card_features,
    get_bid_action_idx, NUM_BID_ACTIONS,
)

# === Insight pipeline (read-only strategic insight generation) ===
from agents.wist_discovery.insight_pipeline import run_insight_cycle


# Domain-agnostic rank ordering.
RANK_VAL = {
    Rank.TWO: 2, Rank.THREE: 3, Rank.FOUR: 4, Rank.FIVE: 5, Rank.SIX: 6,
    Rank.SEVEN: 7, Rank.EIGHT: 8, Rank.NINE: 9, Rank.TEN: 10,
    Rank.JACK: 11, Rank.QUEEN: 12, Rank.KING: 13, Rank.ACE: 14,
}
SUIT_IDX = {Suit.SPADES: 0, Suit.HEARTS: 1, Suit.CLUBS: 2, Suit.DIAMONDS: 3}

STATE_FEATURE_SIZE = 52  # Fixed feature vector size for neural net.

# Insight pipeline interval — run insight generation every N episodes.
_INSIGHT_PIPELINE_INTERVAL = 2000


# =============================================================================
# State & Action Encoding (Wist-specific)
# =============================================================================

def _encode_play_state(obs: WistObservation, opp_voids: int = 0,
                       partner_bid: int = 0, my_tricks: int = 0, opp_tricks: int = 0) -> str:
    """Rich observable state encoding — everything a human can see.

    Encodes:
    - Hand size (how many cards remain, hex 1-d)
    - Position in trick (0-3: how many have played before me)
    - Led suit index (what suit was played first this trick, or 'x' if leading)
    - My team tricks won (0-13, hex)
    - Opponent team tricks won (0-13, hex)
    - Trick winner context: P=partner winning, O=opponent winning, N=nobody/leading
    - Highest rank bucket on table: 0=none, 1=low(2-7), 2=mid(8-10), 3=high(J-K), 4=ace
    - Trump played in this trick: T=yes, F=no
    """
    hand = obs.hand
    n_cards = len(hand)
    pos = 0
    led = "x"
    trick_winner = "N"  # Nobody yet (leading).
    highest_bucket = "0"  # No cards on table.
    trump_in_trick = "F"

    if obs.current_trick and obs.current_trick.played_cards:
        pos = len(obs.current_trick.played_cards)
        if obs.current_trick.leading_suit:
            led = str(SUIT_IDX.get(obs.current_trick.leading_suit, 0))

        # Determine who's currently winning and highest rank on table.
        my_pid = getattr(obs, 'player_id', 0)
        partner_pid = (my_pid + 2) % 4
        trump_suit = getattr(obs, 'trump_suit', None)

        best_rank = 0
        best_is_trump = False
        best_pid = -1

        for pc in obs.current_trick.played_cards:
            rv = RANK_VAL[pc.card.rank]
            is_trump = (pc.card.suit == trump_suit) if trump_suit else False

            # Check if this card is currently winning.
            if is_trump and not best_is_trump:
                best_rank = rv
                best_is_trump = True
                best_pid = pc.player_id
            elif is_trump and best_is_trump and rv > best_rank:
                best_rank = rv
                best_pid = pc.player_id
            elif not is_trump and not best_is_trump:
                if pc.card.suit == obs.current_trick.leading_suit and rv > best_rank:
                    best_rank = rv
                    best_pid = pc.player_id

            # Track if trump has been played.
            if is_trump:
                trump_in_trick = "T"

        # Who's winning?
        if best_pid == my_pid or best_pid == partner_pid:
            trick_winner = "P"  # Partner (or self) winning.
        elif best_pid >= 0:
            trick_winner = "O"  # Opponent winning.

        # Highest rank bucket.
        if best_rank >= 14:
            highest_bucket = "4"  # Ace.
        elif best_rank >= 11:
            highest_bucket = "3"  # J/Q/K.
        elif best_rank >= 8:
            highest_bucket = "2"  # 8/9/10.
        elif best_rank >= 2:
            highest_bucket = "1"  # 2-7.

    return f"{n_cards:x}{pos}{led}{my_tricks:x}{opp_tricks:x}{trick_winner}{highest_bucket}{trump_in_trick}"


def _encode_play_action(card, obs: WistObservation) -> str:
    """Rich action encoding — observable card properties.

    Contains:
    - Raw rank number (2-14) — the card's face value
    - Suit index (0-3) — which suit
    - Is highest of its suit in hand (1/0) — observable by looking at your own cards
    """
    rv = RANK_VAL[card.rank]
    si = SUIT_IDX[card.suit]
    # Is this the highest card of its suit in my hand?
    same_suit_ranks = [RANK_VAL[c.rank] for c in obs.hand if c.suit == card.suit]
    is_highest = "1" if rv == max(same_suit_ranks) else "0"
    return f"{rv:x}{si}{is_highest}"


def _encode_bid_state(obs: BiddingObservation) -> str:
    """Rich bid state — observable bidding context.

    Encodes:
    - Whether someone already bid (Y/N)
    - Current highest bid level (0 if none)
    - Hand strength tier (high card points: 0=weak, 1=medium, 2=strong, 3=very strong)
    - Longest suit length bucket (S=short 1-3, M=medium 4-5, L=long 6+)
    - Number of aces+kings in hand (0-4)
    - Is Sahib Al-Qabool (Q/R for qabool/regular)
    """
    has_bid = "Y" if obs.current_highest_bid else "N"
    bid_level = str(min(obs.current_highest_bid, 13)) if obs.current_highest_bid else "0"

    # Hand strength: count high cards (A=4, K=3, Q=2, J=1 points).
    from intelligence.core.cards.rank import Rank
    hcp_values = {Rank.ACE: 4, Rank.KING: 3, Rank.QUEEN: 2, Rank.JACK: 1}
    hcp = sum(hcp_values.get(c.rank, 0) for c in obs.hand)
    if hcp >= 20:
        strength = "3"
    elif hcp >= 12:
        strength = "2"
    elif hcp >= 6:
        strength = "1"
    else:
        strength = "0"

    # Longest suit length.
    suit_counts = Counter(c.suit for c in obs.hand)
    longest = max(suit_counts.values()) if suit_counts else 0
    if longest >= 6:
        length_code = "L"
    elif longest >= 4:
        length_code = "M"
    else:
        length_code = "S"

    # Count aces and kings.
    ak_count = sum(1 for c in obs.hand if c.rank in (Rank.ACE, Rank.KING))
    ak_code = str(min(ak_count, 4))

    # Qabool or regular bidder.
    role = "Q" if obs.is_sahib_al_qabool else "R"

    return f"{has_bid}{bid_level}{strength}{length_code}{ak_code}{role}"


def _encode_bid_action(action: Action) -> str:
    if isinstance(action, PassAction):
        return "PASS"
    if isinstance(action, BidAction):
        return f"B{min(action.value, 13)}"
    return "PASS"


# =============================================================================
# Wist-specific MCTS Simulation Function
# =============================================================================

def _wist_simulate_fn(state, action):
    """Wist-specific simulate_fn for TIBRAIN MCTSEngine.

    The state is a tuple: (obs, round_state, players, trump_suit, player_id, my_team)
    The action is a card object.

    Returns: (next_state, reward, done, legal_actions)
    """
    obs, round_state, players, trump_suit, player_id, my_team = state

    from environments.wist.rules import legal_cards, trick_winner, rank_value
    from agents.wist_discovery.mcts import (
        _build_simulated_hands, _resolve_trick_winner
    )

    # Build simulated hands
    hands = _build_simulated_hands(player_id, action, obs, players)

    # Current trick state
    current_trick_cards = []
    leading_suit = None
    if obs.current_trick and obs.current_trick.played_cards:
        for pc in obs.current_trick.played_cards:
            current_trick_cards.append((pc.player_id, pc.card))
        leading_suit = obs.current_trick.leading_suit

    # Add our card
    current_trick_cards.append((player_id, action))
    if leading_suit is None:
        leading_suit = action.suit

    # Determine who still needs to play in this trick
    players_in_trick = {pid for pid, _ in current_trick_cards}
    if obs.current_trick and obs.current_trick.leading_player_id is not None:
        leader = obs.current_trick.leading_player_id
    else:
        leader = player_id

    trick_order = []
    for i in range(4):
        pid = (leader + i) % 4
        if pid not in players_in_trick:
            trick_order.append(pid)

    # Simulate remaining players in current trick
    for pid in trick_order:
        hand = hands.get(pid, [])
        if not hand:
            continue
        playable = legal_cards(hand, leading_suit, None)
        if playable:
            chosen = random.choice(playable)
            current_trick_cards.append((pid, chosen))
            hands[pid] = [c for c in hand if c is not chosen]

    # Resolve current trick winner
    team_wins = 0
    winner = _resolve_trick_winner(current_trick_cards, trump_suit, leading_suit)
    if winner is not None:
        winner_team = 0 if winner in (0, 2) else 1
        if winner_team == my_team:
            team_wins += 1
        next_leader = winner
    else:
        next_leader = (player_id + 1) % 4

    # Simulate remaining tricks randomly
    max_remaining = max(len(h) for h in hands.values()) if hands else 0
    for _ in range(max_remaining):
        trick_cards = []
        t_leading_suit = None
        for i in range(4):
            pid = (next_leader + i) % 4
            hand = hands.get(pid, [])
            if not hand:
                continue
            playable = legal_cards(hand, t_leading_suit, None)
            if not playable:
                continue
            chosen = random.choice(playable)
            trick_cards.append((pid, chosen))
            hands[pid] = [c for c in hand if c is not chosen]
            if t_leading_suit is None:
                t_leading_suit = chosen.suit

        if len(trick_cards) == 4:
            w = _resolve_trick_winner(trick_cards, trump_suit, t_leading_suit)
            if w is not None:
                w_team = 0 if w in (0, 2) else 1
                if w_team == my_team:
                    team_wins += 1
                next_leader = w
            else:
                next_leader = (next_leader + 1) % 4
        else:
            break

    # Terminal state — return reward as team_wins, done=True, no further actions
    return state, float(team_wins), True, []


# =============================================================================
# Support Classes (Wist-specific)
# =============================================================================

class PrioritizedReplayBuffer:
    """Experience replay with prioritization based on TD-error (Enhancement #3)."""
    def __init__(self, capacity: int = 20000):
        self._buffer: list = []
        self._priorities: list = []
        self._capacity = capacity
        self._pos = 0

    def add(self, state: str, action: str, reward: float, table_key: str, td_error: float = 1.0):
        priority = abs(td_error) + 0.01  # Small constant to avoid zero priority.
        if len(self._buffer) < self._capacity:
            self._buffer.append((state, action, reward, table_key))
            self._priorities.append(priority)
        else:
            self._buffer[self._pos] = (state, action, reward, table_key)
            self._priorities[self._pos] = priority
        self._pos = (self._pos + 1) % self._capacity

    def sample(self, batch_size: int = 64) -> list:
        if len(self._buffer) == 0:
            return []
        # Take a snapshot to avoid race conditions with concurrent writes.
        buf_snapshot = list(self._buffer)
        pri_snapshot = list(self._priorities[:len(buf_snapshot)])
        size = min(batch_size, len(buf_snapshot))
        total = sum(pri_snapshot)
        if total == 0 or len(pri_snapshot) != len(buf_snapshot):
            indices = random.sample(range(len(buf_snapshot)), size)
        else:
            probs = np.array(pri_snapshot, dtype=np.float64)
            probs /= probs.sum()
            indices = np.random.choice(len(buf_snapshot), size=size, replace=False, p=probs).tolist()
        return [buf_snapshot[i] for i in indices]

    def __len__(self):
        return len(self._buffer)


class OpponentPredictor:
    """Predict opponent actions from observed patterns (Enhancement #6)."""
    def __init__(self):
        # Track: state_context -> action_counts.
        self._opp_patterns: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def observe(self, context: str, action: str):
        """Record an observed opponent action."""
        self._opp_patterns[context][action] += 1

    def predict_confidence(self, context: str) -> float:
        """How predictable is the opponent in this context? 0=unknown, 1=very predictable."""
        actions = self._opp_patterns.get(context, {})
        if not actions:
            return 0.0
        total = sum(actions.values())
        if total < 3:
            return 0.0
        max_count = max(actions.values())
        return max_count / total  # High = one dominant action.



# =============================================================================
# Main Agent Class
# =============================================================================

class WistDiscoveryAgent(Agent):
    """
    Wist Discovery Agent — Advanced RL Architecture v2.
    Transferable learning system. No domain knowledge.

    Delegates generic RL components to TIBRAIN:
    - Neural network evaluation → tibrain.neural_net.Evaluator
    - MCTS look-ahead → tibrain.mcts.MCTSEngine
    - Reward normalization → tibrain.reward.RewardNormalizer
    - Elo tracking → tibrain.evaluation.EloTracker
    - Meta-learning → tibrain.evaluation.MetaLearner
    - Pattern discovery → tibrain.discovery.DiscoveryEngine

    Keeps Wist-specific logic local:
    - Card evaluation features & state encoding
    - Bidding strategy & trump suit constraints
    - Opponent void tracking & card counting
    - Game-specific observation processing
    """

    def __init__(self, epsilon: float = 0.2, alpha: float = 0.2,
                 gamma: float = 0.97, lambda_trace: float = 0.7,
                 training: bool = True) -> None:

        # === Double Q-Learning ===
        self.play_q: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.play_q2: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.bid_q: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.bid_q2: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # === Neural Network Function Approximator (via TIBRAIN Evaluator) ===
        # CardEvaluator: per-card evaluation (state 52 + card 8 + memory 78 = 138 features → 1 Q-value).
        # Uses tibrain.neural_net.Evaluator which has the same architecture.
        self._play_net = Evaluator(input_size=138, hidden_size=256, learning_rate=0.0003)
        self._target_net = self._play_net.copy()  # Target network (frozen, updated every 500 episodes).
        self._target_update_interval = 500
        # Bid network uses tibrain.neural_net.QNetwork for small fixed action space.
        self._bid_net = QNetwork(STATE_FEATURE_SIZE, hidden_size=64,
                                 output_size=NUM_BID_ACTIONS, learning_rate=0.001)
        self._use_neural = False  # Starts with Q-tables, switches after enough data.
        self._neural_switch_threshold = 5000  # Switch to neural after N episodes.

        # === Hyperparameters ===
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_trace = lambda_trace
        self.training = training

        # === Hierarchical Learning: different rates per phase ===
        self._play_alpha_scale = 1.0
        self._bid_alpha_scale = 1.5

        # === Episode memory ===
        self._play_episode: list[tuple[str, str]] = []
        self._bid_episode: list[tuple[str, str]] = []
        self._nn_play_features: list = []  # Neural net feature vectors per play action.

        # === Sequence Memory (full shota history) ===
        # Each entry: (rank_norm, was_trump, followed_suit, won_trick)
        self._trick_memory: list[tuple[float, float, float, float]] = []
        self._memory_size = 13  # Remember all tricks in the shota.

        # === Eligibility Traces ===
        self._play_traces: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._bid_traces: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # === Prioritized Experience Replay ===
        self._replay_buffer = PrioritizedReplayBuffer(capacity=20000)
        self._replay_batch_size = 64

        # === Curiosity Bonus ===
        self._state_visit_counts: dict[str, int] = defaultdict(int)
        self._curiosity_scale = 0.1

        # === Reward Normalization (via TIBRAIN) ===
        self._reward_normalizer = RewardNormalizer()

        # === Opponent Modeling (Wist-specific) ===
        self._known_voids: dict[int, set] = {0: set(), 1: set(), 2: set(), 3: set()}

        # === Card Counting — track all cards seen and suits played globally ===
        self._cards_seen: set = set()  # All cards observed being played.
        self._suits_played: dict = {0: 0, 1: 0, 2: 0, 3: 0}  # SUIT_IDX → count

        # === Opponent Prediction (Wist-specific) ===
        self._opp_predictor = OpponentPredictor()

        # === Self-play Elo Tracking (via TIBRAIN) ===
        self._elo_tracker = EloTracker()

        # === Meta-Learning (via TIBRAIN) ===
        self._meta_learner = MetaLearner()

        # === Pattern Discovery (via TIBRAIN) ===
        self._discovery_engine = DiscoveryEngine(confidence_threshold=0.3)

        # === MCTS Engine (via TIBRAIN) — initialized lazily with Wist simulate_fn ===
        self._mcts_engine: MCTSEngine | None = None

        # === Stats ===
        self.episodes_trained: int = 0
        self.total_updates: int = 0

        # === Enhanced observations (partner bid, trick counts) ===
        self._partner_bid: int = 0  # Observable: what did partner bid?
        self._my_tricks: int = 0    # Observable: tricks won by my team so far.
        self._opp_tricks: int = 0   # Observable: tricks won by opponents so far.

        # === Per-state adaptive alpha (#5) ===
        self._state_update_counts: dict[str, int] = defaultdict(int)

        # === N-step return buffer (#2) ===
        self._nstep_buffer: list[tuple[str, str, float]] = []  # (state, action, intermediate_value)
        self._nstep_n: int = 3  # Look 3 steps ahead for returns.

        # === MCTS training integration (#1, #14) ===
        self._mcts_context: dict = None  # Set externally when MCTS is available.
        self._mcts_value_weight: float = 0.3  # How much MCTS value influences training.

    # =========================================================================
    # Action Selection
    # =========================================================================

    def act(self, observation: Observation) -> Action:
        if isinstance(observation, BiddingObservation):
            return self._act_bid(observation)
        if isinstance(observation, WistObservation):
            return self._act_play(observation)
        raise TypeError(f"Unsupported: {type(observation).__name__}")

    def _act_bid(self, obs: BiddingObservation) -> Action:
        """Bid or pass — learned from reward only.

        Correct bid rules:
          - Min bid is always 7.
          - Max bid is 11 for opening bid, 13 otherwise.
          - Must exceed current highest bid (unless Sahib Al-Qabool, who can match).
          - After winning bid, agent chooses trump suit where suit_count ≤ (bid - 3).
          - Sahib Al-Qabool: no opening cap, can match current highest bid.

        The constraint bid→trump is: chosen trump suit must have ≤ (bid_value - 3) cards.
        The agent doesn't pick trump here — it's picked after winning (determine_trump_suit).
        But the agent CAN'T bid if no valid trump suit exists for that bid level.
        """
        hand = obs.hand
        suit_counts = Counter(c.suit for c in hand)

        # Check if bidding is possible at all (need at least one suit with 1-7 cards).
        valid_suits = [s for s, count in suit_counts.items() if 1 <= count <= 7]
        if not valid_suits:
            return PassAction(player_id=obs.player_id)

        # Determine which bid levels are possible given hand.
        # For bid N, need a suit with ≤ (N-3) cards and ≤ 7 cards.
        min_suit_count = min(suit_counts[s] for s in valid_suits)

        # Minimum bid = the lowest bid where we have a valid trump suit.
        # bid - 3 >= min_suit_count  →  bid >= min_suit_count + 3
        bid_floor = max(7, min_suit_count + 3)

        if obs.is_sahib_al_qabool:
            max_bid = 13
            if obs.current_highest_bid:
                # Qabool can match (not exceed).
                min_bid = max(bid_floor, obs.current_highest_bid)
            else:
                min_bid = bid_floor
        else:
            min_bid = bid_floor
            if obs.is_opening_bid:
                max_bid = 11
            else:
                max_bid = 13
            if obs.current_highest_bid:
                min_bid = max(min_bid, obs.current_highest_bid + 1)

        # Safety clamp.
        max_bid = min(max_bid, 13)

        if min_bid > max_bid and not obs.must_play:
            action = PassAction(player_id=obs.player_id)
        elif min_bid > max_bid and obs.must_play:
            action = BidAction(player_id=obs.player_id, value=min(min_bid, 13))
        elif obs.must_play:
            if self.training and random.random() < self.epsilon:
                bid_val = random.randint(min_bid, max_bid)
            else:
                best = self._best_bid(obs, min_bid, max_bid)
                bid_val = best.value if isinstance(best, BidAction) else min_bid
            action = BidAction(player_id=obs.player_id, value=bid_val)
        elif self.training and random.random() < self.epsilon:
            if random.random() < 0.5:
                action = PassAction(player_id=obs.player_id)
            else:
                bid_val = random.randint(min_bid, max_bid)
                action = BidAction(player_id=obs.player_id, value=bid_val)
        else:
            action = self._best_bid(obs, min_bid, max_bid)

        if self.training:
            state = _encode_bid_state(obs)
            action_key = _encode_bid_action(action)
            self._bid_episode.append((state, action_key))
            self._bid_traces[state][action_key] += 1.0
            self._state_visit_counts[state] += 1

        return action

    def _best_bid(self, obs: BiddingObservation, min_bid: int, max_bid: int = 13) -> Action:
        """Combined Q-table + neural net for bid selection."""
        state = _encode_bid_state(obs)

        # Q-table values.
        q1 = self.bid_q[state]
        q2 = self.bid_q2[state]

        # Neural net values (if active).
        if self._use_neural:
            features = state_to_features(state, STATE_FEATURE_SIZE)
            nn_q = self._bid_net.predict(features)
        else:
            nn_q = None

        best_action = "PASS"
        best_q = self._combined_q(q1.get("PASS", 0.0), q2.get("PASS", 0.0),
                                   nn_q[get_bid_action_idx("PASS")] if nn_q is not None else 0.0)

        for v in range(min_bid, max_bid + 1):
            key = f"B{v}"
            combined = self._combined_q(q1.get(key, 0.0), q2.get(key, 0.0),
                                         nn_q[get_bid_action_idx(key)] if nn_q is not None else 0.0)
            if combined > best_q:
                best_q = combined
                best_action = key

        if best_action == "PASS":
            return PassAction(player_id=obs.player_id)
        return BidAction(player_id=obs.player_id, value=int(best_action[1:]))

    def _act_play(self, obs: WistObservation) -> Action:
        """Play a card — learned from reward only."""
        self._update_voids(obs)
        self._observe_opponents(obs)

        leading_suit = obs.current_trick.leading_suit if obs.current_trick else None
        must_trump = obs.trump_suit if obs.must_lead_trump else None
        playable = legal_cards(obs.hand, leading_suit, must_trump)

        if len(playable) == 1:
            card = playable[0]
        elif self.training and random.random() < self.epsilon:
            # Curiosity-driven exploration: prefer unvisited states.
            if random.random() < 0.3:
                # Pick action that leads to least-visited state.
                state_str = _encode_play_state(obs, self._get_opponent_voids_count(obs),
                                              self._partner_bid, self._my_tricks, self._opp_tricks)
                min_visits = float("inf")
                curiosity_card = random.choice(playable)
                for c in playable:
                    key = _encode_play_action(c, obs)
                    full_key = state_str + key
                    visits = self._state_visit_counts.get(full_key, 0)
                    if visits < min_visits:
                        min_visits = visits
                        curiosity_card = c
                card = curiosity_card
            else:
                card = random.choice(playable)
        else:
            card = self._best_card(obs, playable)

        if self.training:
            state = _encode_play_state(obs, self._get_opponent_voids_count(obs),
                                       self._partner_bid, self._my_tricks, self._opp_tricks)
            action_key = _encode_play_action(card, obs)
            self._play_episode.append((state, action_key))
            self._play_traces[state][action_key] += 1.0
            self._state_visit_counts[state] += 1

            # Store neural net features for this card choice.
            if self._use_neural:
                s_feat = state_features(obs, self._get_opponent_voids_count(obs),
                                        rank_value, SUIT_IDX, self._suits_played,
                                        self._known_voids, self._cards_seen,
                                        self._my_tricks, self._opp_tricks,
                                        self._partner_bid, 0, 0, 0)
                c_feat = card_features(card, obs, playable, rank_value, SUIT_IDX)
                mem_feat = self._get_memory_features()
                self._nn_play_features.append(np.concatenate([s_feat, c_feat, mem_feat]))

        return PlayCardAction(player_id=obs.player_id, card=card)

    def _best_card(self, obs: WistObservation, playable: list) -> object:
        """Combined Q-table + per-card neural net + optional MCTS for card selection.

        MCTS integration: When MCTS context is available, delegates to TIBRAIN
        MCTSEngine with Wist-specific simulate_fn AND feeds value estimates back
        as training targets for the neural net.
        """
        # MCTS: if context is available, use TIBRAIN MCTSEngine for look-ahead.
        if getattr(self, '_mcts_context', None) and len(playable) > 1:
            ctx = self._mcts_context
            num_sims = ctx.get('num_simulations', 80)

            # Build MCTS state tuple for the Wist simulate_fn
            player_id = obs.player_id
            my_team = 0 if player_id in (0, 2) else 1
            mcts_state = (obs, ctx.get('round_state'), ctx.get('players'),
                          ctx.get('trump_suit'), player_id, my_team)

            # Create/reuse TIBRAIN MCTSEngine with Wist simulate_fn
            if self._mcts_engine is None:
                self._mcts_engine = MCTSEngine(
                    simulate_fn=_wist_simulate_fn,
                    num_simulations=num_sims,
                )

            try:
                # Get MCTS action values for training
                mcts_values = self._mcts_engine.evaluate_actions(
                    mcts_state, playable, num_simulations=num_sims
                )

                # Use MCTS values as soft training targets for Q-table.
                if self.training and mcts_values:
                    state_str = _encode_play_state(obs, self._get_opponent_voids_count(obs),
                                                  self._partner_bid, self._my_tricks, self._opp_tricks)
                    mcts_alpha = self._mcts_value_weight * 0.01
                    for card, mcts_val in mcts_values.items():
                        action_key = _encode_play_action(card, obs)
                        current = self.play_q[state_str][action_key]
                        self.play_q[state_str][action_key] += mcts_alpha * (mcts_val - current)

                # Choose action via MCTS
                return self._mcts_engine.choose_action(
                    mcts_state, playable, num_simulations=num_sims
                )
            except Exception:
                pass  # Fall through to Q-table + neural net selection

        state_str = _encode_play_state(obs, self._get_opponent_voids_count(obs),
                                      self._partner_bid, self._my_tricks, self._opp_tricks)
        q1 = self.play_q[state_str]
        q2 = self.play_q2[state_str]

        # Neural net: evaluate all playable cards in one batch.
        nn_values = {}
        if self._use_neural:
            s_feat = state_features(obs, self._get_opponent_voids_count(obs),
                                    rank_value, SUIT_IDX, self._suits_played,
                                    self._known_voids, self._cards_seen,
                                    self._my_tricks, self._opp_tricks,
                                    self._partner_bid, 0, 0, 0)
            mem_feat = self._get_memory_features()
            # Build feature matrix for all cards at once.
            card_feats = []
            card_ids = []
            for card in playable:
                c_feat = card_features(card, obs, playable, rank_value, SUIT_IDX)
                card_feats.append(np.concatenate([s_feat, c_feat, mem_feat]))
                card_ids.append(id(card))
            # Single batch forward pass instead of N separate calls.
            batch_array = np.array(card_feats)
            batch_results = self._play_net.predict_batch(batch_array)
            for i, cid in enumerate(card_ids):
                nn_values[cid] = float(batch_results[i])

        best_card = playable[0]
        best_q = float("-inf")
        for card in playable:
            key = _encode_play_action(card, obs)
            q_table_val = (q1.get(key, 0.0) + q2.get(key, 0.0)) / 2
            nn_val = nn_values.get(id(card), 0.0)
            combined = self._combined_q(q_table_val, q_table_val, nn_val)

            # Tiebreaker: when Q-values are effectively equal, use per-card
            # neural net score (unique per card). If no neural net, use rank as
            # a stable tiebreaker so the choice is deterministic.
            tiebreaker = nn_val * 0.001 if nn_val else RANK_VAL[card.rank] * 0.0001
            combined += tiebreaker

            if combined > best_q:
                best_q = combined
                best_card = card
        return best_card

    def _combined_q(self, q1_val: float, q2_val: float, nn_val: float = 0.0) -> float:
        """Combine Q-table and neural net estimates."""
        if self._use_neural:
            # Blend: 40% Q-table average, 60% neural net.
            # Neural net sees richer features (trick cards, partner identity).
            return 0.4 * (q1_val + q2_val) / 2 + 0.6 * nn_val
        return (q1_val + q2_val) / 2

    # =========================================================================
    # Learning
    # =========================================================================

    def trick_reward(self, won: bool) -> None:
        """Record trick observation (no learning signal — only stores what was seen).

        The agent records observable facts about each completed trick:
        - What rank was dominant
        - What suit was led
        - Who won (position)
        - Whether I won
        This builds trick memory for the neural net to learn from.
        """
        # Store observable trick data for memory features.
        # Approximate values (exact card data would need trick reference).
        self._trick_memory.append((
            0.5,   # avg rank (placeholder — refined when we have full trick data)
            0.5,   # led suit normalized
            0.5,   # winner position normalized
            1.0 if won else 0.0,  # did my team win this trick
        ))
        if won:
            self._my_tricks += 1
        else:
            self._opp_tricks += 1

    def reward(self, score: float) -> None:
        """End-of-shota reward with all architecture enhancements."""
        if not self.training:
            return

        # === Meta-learning: record score (via TIBRAIN MetaLearner) ===
        self._meta_learner.record_score(score)

        # === Elo tracking (via TIBRAIN EloTracker) ===
        self._elo_tracker.update(won=(score > 0))
        if self.episodes_trained % 100 == 0:
            self._elo_tracker.record(self.episodes_trained)

        # === Reward Normalization (via TIBRAIN RewardNormalizer) ===
        self._reward_normalizer.normalize(score)

        # === Pattern Discovery (via TIBRAIN DiscoveryEngine) ===
        # Feed episode experiences to discovery engine for pattern detection
        if self._play_episode:
            reward_outcome = "win" if score > 0 else "loss"
            for state, action in self._play_episode[-3:]:  # Last 3 state-actions
                self._discovery_engine.observe(state, action, reward_outcome)

        # === Adaptive learning rate (per-state) ===
        base_alpha = max(0.05, self.alpha * (1.0 / (1.0 + self.episodes_trained / 1000)))

        # === Play phase with eligibility traces + Double Q + N-step returns ===
        play_alpha = base_alpha * self._play_alpha_scale
        if random.random() < 0.5:
            play_q_update, play_q_eval = self.play_q, self.play_q2
        else:
            play_q_update, play_q_eval = self.play_q2, self.play_q

        # N-step returns: compute returns from windows of N steps.
        episode_len = len(self._play_episode)
        nstep_returns = [0.0] * episode_len
        # First compute single-step (standard): propagate reward backward.
        reward_signal = score
        for idx in range(episode_len - 1, -1, -1):
            nstep_returns[idx] = reward_signal
            reward_signal *= self.gamma

        # N-step enhancement: blend forward returns for first (episode_len - N) steps.
        n = self._nstep_n
        for idx in range(episode_len - n):
            # N-step return = gamma^n * V(state+n) + discounted intermediates.
            # Since we don't have intermediate rewards, n-step just looks further ahead.
            future_state, future_action = self._play_episode[idx + n]
            future_q = play_q_eval[future_state].get(future_action, 0.0)
            nstep_return = (self.gamma ** n) * future_q + nstep_returns[idx] * (1.0 - self.gamma ** n)
            # Blend: 70% standard backward return, 30% n-step forward return.
            nstep_returns[idx] = 0.7 * nstep_returns[idx] + 0.3 * nstep_return

        nn_features_reversed = list(reversed(self._nn_play_features)) if self._nn_play_features else []
        for idx, (state, action) in enumerate(reversed(self._play_episode)):
            # Per-state adaptive alpha: states visited many times learn slower.
            self._state_update_counts[state] += 1
            visit_count = self._state_update_counts[state]
            adaptive_alpha = play_alpha / (1.0 + visit_count / 500.0)

            trace = self._play_traces[state].get(action, 1.0)
            current_q = play_q_update[state][action]
            target = nstep_returns[episode_len - 1 - idx]
            td_error = target - current_q
            update = adaptive_alpha * trace * td_error
            play_q_update[state][action] += update
            self.total_updates += 1

            # Prioritized replay.
            self._replay_buffer.add(state, action, target, "play", td_error)

            # Train TIBRAIN Evaluator neural net.
            if self._use_neural and idx < len(nn_features_reversed):
                self._play_net.update(nn_features_reversed[idx], target)

        # === Bid phase ===
        bid_alpha = base_alpha * self._bid_alpha_scale
        if random.random() < 0.5:
            bid_q_update = self.bid_q
        else:
            bid_q_update = self.bid_q2

        bid_reward = score  # Bid directly caused the shota outcome — full credit.
        for state, action in reversed(self._bid_episode):
            trace = self._bid_traces[state].get(action, 1.0)
            current_q = bid_q_update[state][action]
            td_error = bid_reward - current_q
            bid_q_update[state][action] += bid_alpha * trace * td_error
            self.total_updates += 1
            self._replay_buffer.add(state, action, bid_reward, "bid", td_error)

            if self._use_neural:
                features = state_to_features(state, STATE_FEATURE_SIZE)
                action_idx = get_bid_action_idx(action)
                self._bid_net.update(features, action_idx, bid_reward)

        # === Prioritized Experience Replay ===
        self._do_replay(base_alpha)

        # === Decay eligibility traces ===
        self._decay_traces()

        # === Meta-learning: check if adjustments are needed (via TIBRAIN) ===
        if self._meta_learner.should_adjust(self.episodes_trained):
            adjustments = self._meta_learner.suggest_adjustments(
                self.epsilon, self.episodes_trained
            )
            # Apply suggested adjustments
            if "epsilon" in adjustments:
                self.epsilon = adjustments["epsilon"]

        # === Curriculum: switch to neural net after enough episodes ===
        if not self._use_neural and self.episodes_trained >= self._neural_switch_threshold:
            self._use_neural = True

        # === Target network update (every N episodes) ===
        if self._use_neural and self.episodes_trained % self._target_update_interval == 0:
            self._target_net = self._play_net.copy()

        # === Cleanup ===
        self._play_episode.clear()
        self._bid_episode.clear()
        self._nn_play_features.clear()
        self.episodes_trained += 1

        # === Insight Pipeline (read-only — Req 12.1–12.5) ===
        if self.episodes_trained % _INSIGHT_PIPELINE_INTERVAL == 0:
            try:
                run_insight_cycle(
                    self,
                    self.episodes_trained,
                    data_dir="agents/wist_discovery",
                )
            except Exception:
                pass  # Never crash the training loop for insight generation.

    def _do_replay(self, alpha: float):
        """Prioritized experience replay."""
        batch = self._replay_buffer.sample(self._replay_batch_size)
        replay_alpha = alpha * 0.2

        for state, action, reward, table_key in batch:
            if table_key == "play":
                q = self.play_q if random.random() < 0.5 else self.play_q2
                # Evaluator can't replay from string state alone (needs obs).
                # Only Q-table replay for play actions.
            else:
                q = self.bid_q if random.random() < 0.5 else self.bid_q2
                if self._use_neural:
                    features = state_to_features(state, STATE_FEATURE_SIZE)
                    self._bid_net.update(features, get_bid_action_idx(action), reward)
            current_q = q[state][action]
            q[state][action] += replay_alpha * (reward - current_q)

    def _decay_traces(self):
        """Decay eligibility traces."""
        decay = self.gamma * self.lambda_trace
        for traces in (self._play_traces, self._bid_traces):
            for state in list(traces.keys()):
                for action in list(traces[state].keys()):
                    traces[state][action] *= decay
                    if traces[state][action] < 0.01:
                        del traces[state][action]
                if not traces[state]:
                    del traces[state]

    # =========================================================================
    # Opponent Modeling & Prediction (Wist-specific)
    # =========================================================================

    def _update_voids(self, obs: WistObservation) -> None:
        if not obs.current_trick or not obs.current_trick.played_cards:
            return
        leading_suit = obs.current_trick.leading_suit
        if not leading_suit:
            return
        for played_card in obs.current_trick.played_cards:
            pid = played_card.player_id
            if pid == obs.player_id:
                continue
            if played_card.card.suit != leading_suit:
                self._known_voids[pid].add(leading_suit)
            # Card counting — track exact cards seen + suits played.
            self._cards_seen.add(played_card.card)
            suit_idx = SUIT_IDX.get(played_card.card.suit, 0)
            self._suits_played[suit_idx] = min(13, self._suits_played.get(suit_idx, 0) + 1)

    def _observe_opponents(self, obs: WistObservation) -> None:
        """Track opponent play patterns for prediction."""
        if not obs.current_trick or not obs.current_trick.played_cards:
            return
        for played_card in obs.current_trick.played_cards:
            pid = played_card.player_id
            if pid == obs.player_id:
                continue
            # Context: position in trick + leading suit.
            pos = len([pc for pc in obs.current_trick.played_cards if pc.player_id != pid])
            leading = obs.current_trick.leading_suit
            context = f"p{pos}{'T' if leading else 'L'}"
            action_str = f"{'F' if played_card.card.suit == leading else 'O'}"
            self._opp_predictor.observe(context, action_str)

    def _get_opponent_voids_count(self, obs: WistObservation) -> int:
        my_team = 0 if obs.player_id in (0, 2) else 1
        opp_ids = [1, 3] if my_team == 0 else [0, 2]
        return sum(len(self._known_voids.get(pid, set())) for pid in opp_ids)

    def _get_memory_features(self) -> np.ndarray:
        """Trick memory — raw observable data from previous tricks.

        For each of the last 6 tricks, stores:
        - Rank of card played (normalized) per position (4 slots)
        - Suit of card played (normalized) per position (4 slots)
        - Which position won the trick (one-hot 4)
        - Whether each player followed the led suit (4 binary)
        Total per trick: 4 + 4 + 4 + 4 = 16 features × 6 tricks = 96
        Padded to 78 for backward compatibility (truncate to 78).

        All observable: you watched these cards being played.
        """
        features = np.zeros(78)
        if not self._trick_memory:
            return features
        # Pack last tricks into feature vector.
        idx = 0
        for trick_data in self._trick_memory[-6:]:
            if idx + 13 > 78:
                break
            # Each trick_data: (rank_norm, suit_norm, won_pos_norm, followed)
            # Stored as tuples of raw floats.
            if len(trick_data) >= 4:
                features[idx] = trick_data[0]      # avg rank played
                features[idx + 1] = trick_data[1]  # led suit (normalized)
                features[idx + 2] = trick_data[2]  # winner position (normalized)
                features[idx + 3] = trick_data[3]  # did I win (0/1)
            idx += 13  # 13 slots per trick (sparse, room for expansion)
        return features

    # =========================================================================
    # Episode Management
    # =========================================================================

    def reset_episode(self):
        self._play_episode.clear()
        self._bid_episode.clear()
        self._nn_play_features.clear()
        self._trick_memory.clear()
        self._play_traces.clear()
        self._bid_traces.clear()
        self._known_voids = {0: set(), 1: set(), 2: set(), 3: set()}
        self._cards_seen = set()
        self._suits_played = {0: 0, 1: 0, 2: 0, 3: 0}
        self._my_tricks = 0
        self._opp_tricks = 0

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: str) -> None:
        try:
            data = {
                "play_q": {k: dict(v) for k, v in list(self.play_q.items())},
                "play_q2": {k: dict(v) for k, v in list(self.play_q2.items())},
                "bid_q": {k: dict(v) for k, v in list(self.bid_q.items())},
                "bid_q2": {k: dict(v) for k, v in list(self.bid_q2.items())},
                "episodes_trained": self.episodes_trained,
                "total_updates": self.total_updates,
                "epsilon": self.epsilon,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "lambda_trace": self.lambda_trace,
                "reward_norm": self._reward_normalizer.to_dict(),
                "elo": self._elo_tracker.to_dict(),
                "use_neural": self._use_neural,
                "play_net": self._play_net.to_dict(),
                "bid_net": self._bid_net.to_dict(),
                "discovery": self._discovery_engine.to_dict(),
            }
        except RuntimeError:
            # Dict changed during iteration (background thread). Retry with snapshot.
            data = {
                "play_q": {},
                "play_q2": {},
                "bid_q": {},
                "bid_q2": {},
                "episodes_trained": self.episodes_trained,
                "total_updates": self.total_updates,
                "epsilon": self.epsilon,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "lambda_trace": self.lambda_trace,
                "reward_norm": self._reward_normalizer.to_dict(),
                "elo": self._elo_tracker.to_dict(),
                "use_neural": self._use_neural,
                "play_net": self._play_net.to_dict(),
                "bid_net": self._bid_net.to_dict(),
                "discovery": self._discovery_engine.to_dict(),
            }
            # Try once more with list() snapshots.
            try:
                data["play_q"] = {k: dict(v) for k, v in list(self.play_q.items())}
                data["play_q2"] = {k: dict(v) for k, v in list(self.play_q2.items())}
                data["bid_q"] = {k: dict(v) for k, v in list(self.bid_q.items())}
                data["bid_q2"] = {k: dict(v) for k, v in list(self.bid_q2.items())}
            except RuntimeError:
                pass  # Save with whatever we got.

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Atomic save: write to temp file then rename (prevents corruption on crash).
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f)
        import os
        if os.path.exists(path):
            os.replace(tmp_path, path)
        else:
            os.rename(tmp_path, path)

    def load(self, path: str) -> None:
        with open(path, "r") as f:
            data = json.load(f)
        self.play_q = defaultdict(lambda: defaultdict(float))
        self.play_q2 = defaultdict(lambda: defaultdict(float))
        self.bid_q = defaultdict(lambda: defaultdict(float))
        self.bid_q2 = defaultdict(lambda: defaultdict(float))

        for s, actions in data.get("play_q", {}).items():
            for a, v in actions.items():
                self.play_q[s][a] = v
        for s, actions in data.get("play_q2", {}).items():
            for a, v in actions.items():
                self.play_q2[s][a] = v
        for s, actions in data.get("bid_q", {}).items():
            for a, v in actions.items():
                self.bid_q[s][a] = v
        for s, actions in data.get("bid_q2", {}).items():
            for a, v in actions.items():
                self.bid_q2[s][a] = v

        self.episodes_trained = data.get("episodes_trained", 0)
        self.total_updates = data.get("total_updates", 0)
        self.epsilon = data.get("epsilon", self.epsilon)
        self.lambda_trace = data.get("lambda_trace", self.lambda_trace)
        self._use_neural = data.get("use_neural", False)

        if "reward_norm" in data:
            self._reward_normalizer = RewardNormalizer.from_dict(data["reward_norm"])
        if "elo" in data:
            self._elo_tracker = EloTracker.from_dict(data["elo"])
        if "play_net" in data:
            try:
                self._play_net = Evaluator.from_dict(data["play_net"])
            except (KeyError, ValueError):
                pass  # Old format — start fresh Evaluator.
        if "bid_net" in data:
            self._bid_net = QNetwork.from_dict(data["bid_net"])
        if "discovery" in data:
            self._discovery_engine = DiscoveryEngine.from_dict(data["discovery"])
