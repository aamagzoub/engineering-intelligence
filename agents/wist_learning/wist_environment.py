"""
WistEnvironment — TIBRAIN Environment protocol adapter for the Wist card game.

Bridges the existing Wist game infrastructure to the TIBRAIN Environment protocol,
enabling the generic TIBRAIN training loop to train agents on Wist.

The adapter manages a single player's perspective within a 4-player Wist game.
Opponents are driven by a configurable opponent agent factory.

Episode lifecycle:
    reset() → deal cards, run bidding, set up trick play
    get_legal_actions(state) → legal cards for the learning player
    step(action) → play the chosen card, advance game, return reward
    Episode ends when all 13 tricks are played.
"""

from __future__ import annotations

import random as _random
from typing import Any, Callable

from environments.wist.actions import BidAction, PassAction, PlayCardAction
from environments.wist.observation import BiddingObservation, WistObservation
from environments.wist.player import Player
from environments.wist.round import Round
from environments.wist.round_state import RoundState
from environments.wist.rules import legal_cards, trick_winner
from environments.wist.setup import create_standard_players
from environments.wist.tasmiya_engine import TasmiyaEngine
from environments.wist.trick import Trick
from environments.wist.environment import WistEnvironment as _WistGameEnv
from intelligence.core.agent import Agent
from intelligence.core.cards.card import Card
from intelligence.core.cards.suit import Suit

from tibrain import Action as TibrainAction, State as TibrainState


# ---------------------------------------------------------------------------
# State representation (hashable, opaque to TIBRAIN)
# ---------------------------------------------------------------------------


class WistState:
    """
    Hashable state object wrapping a Wist observation snapshot.

    Stores the observation data as a frozen tuple so it can serve as
    a dictionary key for Q-tables. The TIBRAIN Agent never inspects
    this directly — it passes through a StateEncoder for string keying.
    """

    __slots__ = ("_data", "_hash")

    def __init__(
        self,
        player_id: int,
        hand: tuple[Card, ...],
        trump_suit: Suit | None,
        current_trick_cards: tuple[tuple[int, Card], ...],
        team_scores: tuple[tuple[int, int], ...],
        completed_tricks_count: int,
        must_lead_trump: bool,
    ) -> None:
        self._data = (
            player_id,
            hand,
            trump_suit,
            current_trick_cards,
            team_scores,
            completed_tricks_count,
            must_lead_trump,
        )
        self._hash = hash(self._data)

    @property
    def player_id(self) -> int:
        return self._data[0]

    @property
    def hand(self) -> tuple[Card, ...]:
        return self._data[1]

    @property
    def trump_suit(self) -> Suit | None:
        return self._data[2]

    @property
    def current_trick_cards(self) -> tuple[tuple[int, Card], ...]:
        return self._data[3]

    @property
    def team_scores(self) -> tuple[tuple[int, int], ...]:
        return self._data[4]

    @property
    def completed_tricks_count(self) -> int:
        return self._data[5]

    @property
    def must_lead_trump(self) -> bool:
        return self._data[6]

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WistState):
            return NotImplemented
        return self._data == other._data

    def __repr__(self) -> str:
        return (
            f"WistState(player={self.player_id}, "
            f"hand_size={len(self.hand)}, "
            f"tricks={self.completed_tricks_count})"
        )


# ---------------------------------------------------------------------------
# WistAction wrapper (hashable action for TIBRAIN)
# ---------------------------------------------------------------------------


class WistAction:
    """
    Hashable action wrapping a Card to play.

    The underlying card is the action — TIBRAIN treats this as opaque
    and passes it through an ActionEncoder for string keying.
    """

    __slots__ = ("card", "_hash")

    def __init__(self, card: Card) -> None:
        self.card = card
        self._hash = hash(card)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WistAction):
            return NotImplemented
        return self.card == other.card

    def __repr__(self) -> str:
        return f"WistAction({self.card})"


# ---------------------------------------------------------------------------
# Minimal opponent agent for self-contained environment
# ---------------------------------------------------------------------------


class _RandomOpponent(Agent):
    """Minimal random-play agent used as default opponent in the adapter."""

    def act(self, observation: Any) -> Any:
        if isinstance(observation, BiddingObservation):
            return self._act_bidding(observation)

        if isinstance(observation, WistObservation):
            hand = observation.hand
            leading_suit = None
            must_lead_trump = None

            if observation.current_trick:
                leading_suit = observation.current_trick.leading_suit
            if observation.must_lead_trump and observation.trump_suit:
                must_lead_trump = observation.trump_suit

            playable = legal_cards(hand, leading_suit, must_lead_trump)
            card = _random.choice(playable)
            return PlayCardAction(player_id=observation.player_id, card=card)

        raise TypeError(f"Unsupported observation: {type(observation).__name__}")

    def _act_bidding(self, obs: BiddingObservation) -> Any:
        """Simple bidding logic: bid 7 most of the time to avoid all-pass Dak."""
        from collections import Counter

        hand = obs.hand
        suit_counts = Counter(card.suit for card in hand)
        longest = max(suit_counts.values()) if suit_counts else 0

        # Can't bid with 8+ in one suit (Dak territory)
        if longest >= 8:
            return PassAction(player_id=obs.player_id)

        # Determine minimum feasible bid
        min_bid = max(7, longest + 3)

        # If Sahib Al-Qabool and must play (3rd Dak), always bid
        if obs.must_play:
            bid_value = min_bid
            if obs.current_highest_bid and bid_value < obs.current_highest_bid:
                bid_value = obs.current_highest_bid
            return BidAction(player_id=obs.player_id, value=min(bid_value, 13))

        # 60% chance to bid if no existing bid, 40% chance to outbid
        if obs.current_highest_bid is None:
            if _random.random() < 0.6:
                bid_value = min_bid
                # Respect opening bid max 11 rule
                if not obs.is_sahib_al_qabool and obs.is_opening_bid:
                    bid_value = min(bid_value, 11)
                return BidAction(player_id=obs.player_id, value=bid_value)
        else:
            if _random.random() < 0.4:
                bid_value = obs.current_highest_bid + 1
                if obs.is_sahib_al_qabool:
                    bid_value = obs.current_highest_bid  # Qabool can match
                if bid_value <= 13:
                    return BidAction(player_id=obs.player_id, value=bid_value)

        return PassAction(player_id=obs.player_id)


# ---------------------------------------------------------------------------
# TIBRAIN Environment adapter
# ---------------------------------------------------------------------------


class WistEnvironmentAdapter:
    """
    Adapts the Wist card game to the TIBRAIN Environment protocol.

    Conforms to:
        tibrain.Environment.reset() -> State
        tibrain.Environment.observe() -> State
        tibrain.Environment.get_legal_actions(state) -> list[Action]
        tibrain.Environment.step(action) -> tuple[State, float, dict]

    The adapter manages one learning player (default player_id=0, team 0).
    Opponents are controlled by the provided opponent_factory.

    Parameters
    ----------
    learner_player_id : int
        The player seat controlled by the TIBRAIN agent (0-3).
    opponent_factory : Callable[[], Agent] | None
        Factory that creates opponent agents. Defaults to random agents.
    sahib_al_qabool_rotation : bool
        Whether to rotate the Qabool position each episode.
    """

    def __init__(
        self,
        learner_player_id: int = 0,
        opponent_factory: Callable[[], Agent] | None = None,
        sahib_al_qabool_rotation: bool = True,
    ) -> None:
        self._learner_id = learner_player_id
        self._opponent_factory = opponent_factory or _RandomOpponent
        self._rotate_qabool = sahib_al_qabool_rotation
        self._episode_count = 0

        # Game state (set during reset)
        self._players: list[Player] | None = None
        self._round: Round | None = None
        self._game_env: _WistGameEnv | None = None
        self._agents: list[Agent | None] | None = None
        self._team_tricks: dict[int, int] = {0: 0, 1: 0}
        self._current_leader_id: int = 0
        self._tricks_played: int = 0
        self._trump_suit: Suit | None = None
        self._winning_bid_value: int | None = None
        self._playing_team_id: int | None = None
        self._done: bool = True

        # Internal: track whose turn it is within a trick
        self._trick_play_order: list[int] = []
        self._trick_play_index: int = 0

    # ------------------------------------------------------------------
    # TIBRAIN Environment protocol methods
    # ------------------------------------------------------------------

    def reset(self) -> TibrainState:
        """
        Start a new Wist hand (deal, bid, prepare for trick play).

        Returns the initial state from the learner's perspective.
        Skips Dak hands automatically (re-deals until a playable hand).
        """
        self._episode_count += 1
        max_attempts = 50  # Safety limit to avoid infinite loops

        for _ in range(max_attempts):
            state = self._try_setup_hand()
            if state is not None:
                return state

        # If we exhaust attempts (extremely unlikely), return a terminal state
        self._done = True
        return self._build_state()

    def observe(self) -> TibrainState:
        """Return the current game state as a hashable WistState."""
        return self._build_state()

    def get_legal_actions(self, state: TibrainState) -> list[TibrainAction]:
        """
        Return the legal card plays available to the learning player.

        Only called when it's the learner's turn to act.
        """
        if self._done or self._game_env is None:
            return []

        obs = self._game_env.observe(self._learner_id)
        leading_suit = None
        must_lead_trump = None

        if obs.current_trick:
            leading_suit = obs.current_trick.leading_suit
        if obs.must_lead_trump and obs.trump_suit:
            must_lead_trump = obs.trump_suit

        playable = legal_cards(list(obs.hand), leading_suit, must_lead_trump)
        return [WistAction(card) for card in playable]

    def step(self, action: TibrainAction) -> tuple[TibrainState, float, dict]:
        """
        Execute the learner's chosen card, then advance game state.

        Plays opponent cards until it's the learner's turn again or
        the trick/episode ends.

        Returns
        -------
        next_state : WistState
            The state after the action and any subsequent opponent plays.
        reward : float
            Immediate reward signal (per-trick shaping).
        info : dict
            Contains {"done": bool} indicating episode termination,
            plus optional metadata.
        """
        if not isinstance(action, WistAction):
            raise TypeError(f"Expected WistAction, got {type(action).__name__}")

        if self._done:
            return self._build_state(), 0.0, {"done": True}

        # Play the learner's card
        play_action = PlayCardAction(
            player_id=self._learner_id, card=action.card
        )
        self._game_env.apply_action(play_action)
        self._trick_play_index += 1

        # Continue playing opponents until trick completes or it's learner's turn
        reward = self._advance_to_learner_turn()

        next_state = self._build_state()
        info = {
            "done": self._done,
            "tricks_played": self._tricks_played,
            "team_tricks": dict(self._team_tricks),
        }

        return next_state, reward, info

    # ------------------------------------------------------------------
    # Internal game management
    # ------------------------------------------------------------------

    def _try_setup_hand(self) -> WistState | None:
        """
        Attempt to set up a playable hand. Returns None if Dak occurs.
        """
        self._players = create_standard_players()
        self._round = Round(self._players)
        self._round.deal()

        # Skip card-based Dak hands
        if self._round.has_card_based_dak():
            return None

        # Create opponent agents (placeholder for learner position)
        self._agents = [None] * 4
        for pid in range(4):
            if pid != self._learner_id:
                self._agents[pid] = self._opponent_factory()

        # Run bidding (using a simple bidding agent for the learner)
        bidding_agents: list[Agent] = []
        learner_bid_agent = _RandomOpponent()
        for pid in range(4):
            if pid == self._learner_id:
                bidding_agents.append(learner_bid_agent)
            else:
                bidding_agents.append(self._agents[pid])

        qabool_id = (
            self._episode_count % 4 if self._rotate_qabool else 0
        )
        tasmiya = TasmiyaEngine()
        result = tasmiya.run(
            players=self._players,
            agents=bidding_agents,
            sahib_al_qabool_id=qabool_id,
        )

        if result.is_dak:
            return None

        # Set up for trick play
        round_state = self._round.state
        round_state.trump_suit = result.trump_suit
        round_state.winning_bidder_id = result.winning_bidder_id

        self._game_env = _WistGameEnv(round_state)
        self._trump_suit = result.trump_suit
        self._winning_bid_value = result.winning_bid_value
        self._playing_team_id = result.playing_team_id
        self._current_leader_id = result.winning_bidder_id
        self._team_tricks = {0: 0, 1: 0}
        self._tricks_played = 0
        self._done = False

        # Start the first trick
        self._start_new_trick()

        # Advance through any opponents before the learner's first turn
        self._advance_opponents_to_learner()

        return self._build_state()

    def _start_new_trick(self) -> None:
        """Initialize a new trick with the current leader."""
        round_state = self._round.state
        round_state.current_trick = Trick(leading_player_id=self._current_leader_id)
        self._trick_play_order = [
            (self._current_leader_id + i) % 4 for i in range(4)
        ]
        self._trick_play_index = 0

    def _advance_opponents_to_learner(self) -> None:
        """
        Play opponents' cards from the start of the trick until it's
        the learner's turn. Used after reset/new trick when the learner
        may not be the first to play.
        """
        while (
            self._trick_play_index < 4
            and self._trick_play_order[self._trick_play_index] != self._learner_id
        ):
            self._play_opponent_card()

    def _advance_to_learner_turn(self) -> float:
        """
        After the learner plays, continue advancing until:
        1. The current trick completes (award trick reward)
        2. Start new trick if needed
        3. Advance opponents until learner's turn in new trick
        4. Or episode ends

        Returns accumulated reward.
        """
        total_reward = 0.0

        # Complete current trick — play remaining opponents after learner
        while self._trick_play_index < 4:
            current_player = self._trick_play_order[self._trick_play_index]
            if current_player == self._learner_id:
                # Edge case: learner appears twice (shouldn't happen, but be safe)
                break
            self._play_opponent_card()

        # Trick is complete — resolve it
        if self._trick_play_index >= 4:
            total_reward += self._resolve_trick()

        # If episode not done, start new trick and advance to learner's turn
        if not self._done:
            self._start_new_trick()
            self._advance_opponents_to_learner()

        return total_reward

    def _play_opponent_card(self) -> None:
        """Have the current opponent play a card."""
        if self._trick_play_index >= 4:
            return

        player_id = self._trick_play_order[self._trick_play_index]
        agent = self._agents[player_id]

        obs = self._game_env.observe(player_id)
        action = agent.act(obs)
        self._game_env.apply_action(action)
        self._trick_play_index += 1

    def _resolve_trick(self) -> float:
        """
        Determine the trick winner, update scores, check for episode end.

        Returns the reward signal for this trick.
        """
        round_state = self._round.state
        completed_trick = round_state.current_trick

        winner = trick_winner(completed_trick, self._trump_suit)
        round_state.completed_tricks.append(completed_trick)
        round_state.current_trick = None

        winner_team = self._players[winner].team_id
        self._team_tricks[winner_team] += 1
        self._tricks_played += 1
        self._current_leader_id = winner

        # Check if episode is done (all 13 tricks played)
        if self._tricks_played >= 13:
            self._done = True
            return self._compute_terminal_reward(winner_team)

        # Per-trick reward shaping
        return self._compute_trick_reward(winner_team)

    def _compute_trick_reward(self, winner_team: int) -> float:
        """
        Compute per-trick reward signal.

        Positive reward when the learner's team wins a trick,
        negative when the opponent team wins.
        """
        learner_team = self._players[self._learner_id].team_id

        if winner_team == learner_team:
            # Bonus for maintaining seek potential
            opp_team = 1 - learner_team
            if self._team_tricks[opp_team] == 0:
                seek_bonus = 0.05 * self._team_tricks[learner_team]
                return 0.3 + seek_bonus
            return 0.25
        else:
            # Penalty, extra harsh if seek was broken
            learner_team_tricks = self._team_tricks.get(learner_team, 0)
            opp_team_tricks = self._team_tricks.get(1 - learner_team, 0)
            if opp_team_tricks == 1 and learner_team_tricks >= 5:
                return -0.5  # Lost seek after a long streak
            return -0.15

    def _compute_terminal_reward(self, last_trick_winner_team: int) -> float:
        """
        Compute end-of-episode reward combining trick outcome and shota result.

        Includes:
        - Last trick reward
        - Win/loss bonus for the whole shota
        - Seek bonus/penalty
        - Bid fulfillment bonus
        """
        learner_team = self._players[self._learner_id].team_id
        opp_team = 1 - learner_team

        # Last trick reward
        trick_reward = self._compute_trick_reward(last_trick_winner_team)

        # Shota outcome
        my_tricks = self._team_tricks[learner_team]
        opp_tricks = self._team_tricks[opp_team]
        team_won = my_tricks > opp_tricks

        if team_won:
            shota_reward = 1.0
        else:
            shota_reward = -1.0

        # Seek bonus
        if my_tricks == 13:
            shota_reward += 3.0  # Achieved seek
        elif opp_tricks == 13:
            shota_reward -= 2.0  # Got seeked

        # Bid fulfillment bonus
        if (
            self._playing_team_id == learner_team
            and self._winning_bid_value is not None
        ):
            if my_tricks >= self._winning_bid_value:
                shota_reward += 0.5  # Met the bid
            else:
                shota_reward -= 0.5  # Failed the bid

        return trick_reward + shota_reward

    def _build_state(self) -> WistState:
        """
        Construct a hashable WistState from current game state.
        """
        if self._done or self._game_env is None:
            # Terminal or uninitialized state
            return WistState(
                player_id=self._learner_id,
                hand=(),
                trump_suit=None,
                current_trick_cards=(),
                team_scores=tuple(sorted(self._team_tricks.items())),
                completed_tricks_count=self._tricks_played,
                must_lead_trump=False,
            )

        obs = self._game_env.observe(self._learner_id)

        # Current trick cards as tuple of (player_id, card)
        trick_cards: tuple[tuple[int, Card], ...] = ()
        if obs.current_trick and obs.current_trick.played_cards:
            trick_cards = tuple(
                (pc.player_id, pc.card)
                for pc in obs.current_trick.played_cards
            )

        return WistState(
            player_id=self._learner_id,
            hand=tuple(sorted(obs.hand, key=lambda c: (c.suit.name, c.rank.value))),
            trump_suit=obs.trump_suit,
            current_trick_cards=trick_cards,
            team_scores=tuple(sorted(self._team_tricks.items())),
            completed_tricks_count=self._tricks_played,
            must_lead_trump=obs.must_lead_trump,
        )
