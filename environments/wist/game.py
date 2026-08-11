"""
Full Wist game orchestrator.

A Wist game consists of up to 5 Shotas. The first team to reach
25 points wins. Seek (winning all 13 tricks) ends the game immediately.

This module handles:
- Sahib Al-Qabool rotation (counter-clockwise each Shota)
- Card-based Dak detection and re-deal
- Pass-based Dak with the 2-per-game limit
- Scoring after each Shota
- Game termination conditions
"""

from dataclasses import dataclass, field

from agents.random.random_agent import RandomAgent
from environments.wist.dak import triggers_card_based_dak
from environments.wist.environment import WistEnvironment
from environments.wist.game_state import DakType, GameState
from environments.wist.playing_engine import PlayingEngine
from environments.wist.round import Round
from environments.wist.scoring import detect_seek, score_shota
from environments.wist.setup import create_standard_players
from environments.wist.tasmiya_engine import TasmiyaEngine, TasmiyaResult, determine_trump_suit
from environments.wist.player import Player
from intelligence.core.agent import Agent


MAX_REDEAL_ATTEMPTS = 10


@dataclass
class ShotaOutcome:
    """Result of one completed Shota within a game."""

    shota_number: int
    playing_team_id: int
    defending_team_id: int
    bid: int
    trump_suit_name: str
    winning_bidder_id: int
    team_tricks: dict[int, int]
    score_delta: dict[int, int]
    seek_team_id: int | None = None
    was_dak: bool = False
    dak_type: str | None = None


@dataclass
class GameResult:
    """Final result of a complete Wist game."""

    winner_team_id: int | None
    final_scores: dict[int, int]
    total_shotas: int
    shota_outcomes: list[ShotaOutcome] = field(default_factory=list)
    ended_by_seek: bool = False


class WistGame:
    """
    Orchestrates a full Wist game (up to 5 Shotas).

    Usage:
        game = WistGame(players, agents)
        result = game.play()
    """

    def __init__(
        self,
        players: list[Player] | None = None,
        agents: list[Agent] | None = None,
    ) -> None:
        self.players = players or create_standard_players()
        self.agents = agents or [RandomAgent() for _ in range(4)]

        self.game_state = GameState()
        self.tasmiya_engine = TasmiyaEngine()
        self.playing_engine = PlayingEngine()

        # Sahib Al-Qabool starts at player 0 and rotates counter-clockwise.
        self.sahib_al_qabool_id = 0

        self.shota_outcomes: list[ShotaOutcome] = []
        self.events: list[str] = []

    def play(self) -> GameResult:
        """
        Play a full game and return the result.
        """

        while not self.game_state.is_finished():
            outcome = self._play_one_shota()

            if outcome is not None:
                self.shota_outcomes.append(outcome)

            if self.game_state.is_finished():
                break

        return GameResult(
            winner_team_id=self.game_state.winner_team_id,
            final_scores=dict(self.game_state.team_scores),
            total_shotas=self.game_state.completed_shotas,
            shota_outcomes=list(self.shota_outcomes),
            ended_by_seek=(
                self.game_state.winner_team_id is not None
                and any(o.seek_team_id is not None for o in self.shota_outcomes)
            ),
        )

    def _play_one_shota(self) -> ShotaOutcome | None:
        """
        Play one Shota: deal → check Dak → Tasmiya → play 13 tricks → score.
        Returns a ShotaOutcome, or None if it was a card-based Dak in the first Shota.
        """

        shota_number = self.game_state.completed_shotas + 1

        # --- Deal and check card-based Dak ---
        round_ = Round(self.players)
        round_.deal()

        redeal_count = 0
        while round_.has_card_based_dak() and redeal_count < MAX_REDEAL_ATTEMPTS:
            dak_player = round_.first_card_based_dak_player_id()
            self.events.append(
                f"Shota {shota_number}: Card-based Dak by Player {dak_player + 1}. Re-dealing."
            )

            # Card-based Dak: same Qabool, re-deal.
            # First Shota card-based Dak does NOT count toward 5 Shotas.
            if shota_number > 1:
                self.game_state.apply_dak(DakType.CARD_BASED)
                return ShotaOutcome(
                    shota_number=shota_number,
                    playing_team_id=-1,
                    defending_team_id=-1,
                    bid=0,
                    trump_suit_name="",
                    winning_bidder_id=-1,
                    team_tricks={0: 0, 1: 0},
                    score_delta={0: 0, 1: 0},
                    was_dak=True,
                    dak_type="card_based",
                )

            # First Shota: re-deal without counting.
            round_ = Round(self.players)
            round_.deal()
            redeal_count += 1

        # --- Al-Tasmiya (bidding) ---
        tasmiya_result = self.tasmiya_engine.run(
            players=self.players,
            agents=self.agents,
            sahib_al_qabool_id=self.sahib_al_qabool_id,
        )

        if tasmiya_result.is_dak:
            # Pass-based Dak.
            self.events.append(
                f"Shota {shota_number}: Pass-based Dak. "
                f"Sahib Al-Qabool: Player {self.sahib_al_qabool_id + 1}."
            )

            # Check if we can still Dak (max 2 per game).
            if self.game_state.pass_based_dak_count < 2:
                self.game_state.apply_dak(DakType.PASS_BASED)
                # Rotate Qabool after pass-based Dak.
                self._rotate_qabool()

                return ShotaOutcome(
                    shota_number=shota_number,
                    playing_team_id=-1,
                    defending_team_id=-1,
                    bid=0,
                    trump_suit_name="",
                    winning_bidder_id=-1,
                    team_tricks={0: 0, 1: 0},
                    score_delta={0: 0, 1: 0},
                    was_dak=True,
                    dak_type="pass_based",
                )
            else:
                # Third time all pass: Qabool must play with bid 7.
                self.events.append(
                    f"Shota {shota_number}: Third pass-based Dak attempt. "
                    f"Forcing Sahib Al-Qabool (Player {self.sahib_al_qabool_id + 1}) to play."
                )
                trump = determine_trump_suit(self.players[self.sahib_al_qabool_id].hand, 7)
                tasmiya_result = TasmiyaResult(
                    winning_bidder_id=self.sahib_al_qabool_id,
                    winning_bid_value=7,
                    trump_suit=trump,
                    playing_team_id=self.players[self.sahib_al_qabool_id].team_id,
                    defending_team_id=(
                        1 if self.players[self.sahib_al_qabool_id].team_id == 0 else 0
                    ),
                    sahib_al_qabool_id=self.sahib_al_qabool_id,
                    is_dak=False,
                    bid_history=[],
                )

        # --- Play the Shota (13 tricks) ---
        round_.state.trump_suit = tasmiya_result.trump_suit
        round_.state.winning_bidder_id = tasmiya_result.winning_bidder_id
        round_.next_leading_player_id = tasmiya_result.winning_bidder_id

        environment = WistEnvironment(round_.state)

        team_tricks = self.playing_engine.play_shota(
            round_=round_,
            environment=environment,
            agents=self.agents,
        )

        # --- Check for Seek ---
        seek_team = detect_seek(team_tricks)
        if seek_team is not None:
            self.game_state.apply_seek(seek_team)
            self.events.append(
                f"Shota {shota_number}: SEEK by Team {seek_team + 1}! Game over."
            )

            self._rotate_qabool()

            return ShotaOutcome(
                shota_number=shota_number,
                playing_team_id=tasmiya_result.playing_team_id,
                defending_team_id=tasmiya_result.defending_team_id,
                bid=tasmiya_result.winning_bid_value,
                trump_suit_name=tasmiya_result.trump_suit.name,
                winning_bidder_id=tasmiya_result.winning_bidder_id,
                team_tricks=dict(team_tricks),
                score_delta={0: 0, 1: 0},
                seek_team_id=seek_team,
            )

        # --- Score the Shota ---
        score_delta = score_shota(
            playing_team_id=tasmiya_result.playing_team_id,
            defending_team_id=tasmiya_result.defending_team_id,
            bid=tasmiya_result.winning_bid_value,
            playing_team_tricks=team_tricks[tasmiya_result.playing_team_id],
            defending_team_tricks=team_tricks[tasmiya_result.defending_team_id],
        )

        self.game_state.apply_shota_score(score_delta)

        self.events.append(
            f"Shota {shota_number}: Team 1 scored {score_delta.get(0, 0)}, "
            f"Team 2 scored {score_delta.get(1, 0)}. "
            f"Totals: {self.game_state.team_scores[0]} / {self.game_state.team_scores[1]}."
        )

        # Rotate Qabool for the next Shota.
        self._rotate_qabool()

        return ShotaOutcome(
            shota_number=shota_number,
            playing_team_id=tasmiya_result.playing_team_id,
            defending_team_id=tasmiya_result.defending_team_id,
            bid=tasmiya_result.winning_bid_value,
            trump_suit_name=tasmiya_result.trump_suit.name,
            winning_bidder_id=tasmiya_result.winning_bidder_id,
            team_tricks=dict(team_tricks),
            score_delta=dict(score_delta),
        )

    def _rotate_qabool(self) -> None:
        """Rotate Sahib Al-Qabool counter-clockwise."""
        self.sahib_al_qabool_id = (self.sahib_al_qabool_id + 1) % 4
