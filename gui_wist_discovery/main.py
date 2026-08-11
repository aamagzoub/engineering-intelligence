"""
Sudanese Wist -- Discovery Watcher

Watch a Discovery AI learn Wist from scratch with only:
- Environment (there's a game)
- Legal moves (what cards/bids are allowed)
- Score signal (end-of-shota points)

Usage:
    python gui_wist_discovery/main.py
"""

import sys
import os
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame

from gui_wist_discovery.constants import (
    SCREEN_WIDTH, SCREEN_HEIGHT, FPS, TITLE, TRICK_DELAY_MS,
)
from gui_wist_discovery.game_engine import (
    sort_hand, try_setup_shota, play_trick, score_completed_shota,
)
from gui_wist_discovery.milestones import (
    save_milestones, load_milestones, check_milestones,
    auto_discover, create_auto_stats,
)
from gui_wist_discovery.insights import generate_insights
from gui_wist_discovery.training import (
    snapshot_brain, create_opponent, create_training_clone,
    run_background_training, STAGE_STAGNATION_THRESHOLD, STAGE_CONFIG,
)
from gui_wist_discovery.renderer import Renderer

from agents.wist_discovery.discovery_agent import WistDiscoveryAgent
from environments.wist.environment import WistEnvironment


class WistDiscoveryWatcher:
    """Watch the AI discover Wist strategy from scratch."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(TITLE)
        self.clock = pygame.time.Clock()
        self.running = True

        self.fonts = {
            "title": pygame.font.SysFont("Segoe UI", 22, bold=True),
            "large": pygame.font.SysFont("Segoe UI", 15, bold=True),
            "medium": pygame.font.SysFont("Segoe UI", 15),
            "small": pygame.font.SysFont("Segoe UI", 10),
        }
        self.renderer = Renderer(self.screen, self.fonts)

        # Mode & speed.
        self.speed = 3.0
        self.paused = False
        self.state = "idle"

        # Agent — self-play.
        self.model_path = "agents/wist_discovery/wist_discovery_model.json"
        self.discovery = WistDiscoveryAgent(training=True)
        if os.path.exists(self.model_path):
            try:
                self.discovery.load(self.model_path)
            except Exception:
                pass

        # Opponent shares Q-tables (stage 1: self-play).
        self.opp = WistDiscoveryAgent(training=False)
        self._sync_opponent()

        # Game state.
        self.shota_num = 0
        self.game_num = 0
        self.team_scores = [0, 0]
        self.shota_scores = []
        self.trick_num = 0
        self.current_trick_cards = []
        self.last_winner = -1
        self.last_action_time = 0
        self._hands = {}

        # Event log.
        self._event_log = []

        # Milestones.
        achieved, milestones_list, accumulated_compute, session_stats = load_milestones()
        self._milestones_achieved = achieved
        self._milestones_list = milestones_list
        self._accumulated_compute = accumulated_compute
        self._last_discovery_compute = None

        # Restore session stats (persist across close/open).
        self._team1_wins = session_stats.get("team1_wins", 0)
        self._team2_wins = session_stats.get("team2_wins", 0)
        self._bids_met_t1 = session_stats.get("bids_met_t1", 0)
        self._bids_met_t2 = session_stats.get("bids_met_t2", 0)
        self._seeks_t1 = session_stats.get("seeks_t1", 0)
        self._seeks_t2 = session_stats.get("seeks_t2", 0)
        self._opponent_stage = session_stats.get("opponent_stage", 1)
        self._daks_t1 = session_stats.get("daks_t1", 0)
        self._daks_t2 = session_stats.get("daks_t2", 0)
        self._shotas_won_t1 = session_stats.get("shotas_won_t1", 0)
        self._shotas_won_t2 = session_stats.get("shotas_won_t2", 0)

        # Timing.
        self._app_start_time = time.monotonic()
        self._total_paused_time = 0.0
        self._pause_start = None

        # Stats — loaded from session_stats (persist across close/open).
        self.shotas_played = session_stats.get("shotas_played", 0)
        self.seeks_achieved = session_stats.get("seeks_achieved", 0)
        self.bids_met = session_stats.get("bids_met", 0)
        self.bids_failed = session_stats.get("bids_failed", 0)
        self._wist_win_history = []
        self._last_score = None
        self._best_score = session_stats.get("best_score", -999)
        self._bids_met_streak = session_stats.get("bids_met_streak", 0)
        self._bid_history = []
        self._score_at_shota3 = None

        # Auto-discovery stats.
        self._auto_stats = create_auto_stats()

        # Opponent curriculum.
        # _opponent_stage is loaded from session_stats above.
        self._last_discovery_episode_for_stage = 0
        self._frozen_snapshot = None
        self._best_snapshot = None

        # Insights cache.
        self._cached_insights = []
        self._last_insight_episode = 0

        # Scroll offsets.
        self._disc_scroll_offset = 0
        self._insight_scroll_offset = 0

        # Info overlay toggle.
        self._show_info = False

        # Insight category filter (None = show all).
        self._insight_filter = None
        self._insight_chip_rects = {}  # {category: pygame.Rect} — set by renderer.
        self._confidence_filter = None  # Minimum confidence to show (None = no filter).

        # Background training.
        self._bg_active = False
        self._bg_agent = None

        # Button rects (set by renderer).
        self._mode_btn_rect = None
        self._reset_btn_rect = None

        self._log(f"Wist Discovery Agent: {self.discovery.episodes_trained} shotas learned")
        self._start_new_game()

    # ─── Event Log ──────────────────────────────────────────────────────────────

    def _log(self, msg):
        self._event_log.append(msg)
        if len(self._event_log) > 200:
            self._event_log = self._event_log[-200:]

    # ─── Game Flow ──────────────────────────────────────────────────────────────

    def _start_new_game(self):
        self.game_num += 1
        self.shota_num = 0
        self.team_scores = [0, 0]
        self.shota_scores = []
        self._log(f"{'=' * 30} Game #{self.game_num} {'=' * 30}")
        self._start_new_shota()

    def _start_new_shota(self):
        self.shota_num += 1
        if self.shota_num > 5:
            self._end_game()
            return

        self.trick_num = 0
        self.current_trick_cards = []
        self._log(f"--- Shota {self.shota_num}/5 ---")

        agents = [self.discovery, self.opp]
        players, rnd, result, logs = try_setup_shota(agents, self.shota_num, self.game_num)

        for msg in logs:
            self._log(msg)

        if players is None:
            # Dak — doesn't count as a played shota. Decrement back.
            self.shota_num -= 1
            self.discovery.reset_episode()
            # Track Dak per team (alternate based on qabool rotation).
            qabool_id = (self.shota_num) % 4
            if qabool_id in (0, 2):
                self._daks_t1 += 1
            else:
                self._daks_t2 += 1
            self.state = "scoring"
            self.last_action_time = pygame.time.get_ticks()
            return

        self._tasmiya_result = result
        self._players = players
        self._agents_list = [self.discovery, self.opp, self.discovery, self.opp]
        self._round = rnd
        self._log(f"  Bid: {result.winning_bid_value} by P{result.winning_bidder_id}")

        # Setup play.
        rnd.state.trump_suit = result.trump_suit
        rnd.state.winning_bidder_id = result.winning_bidder_id
        rnd.next_leading_player_id = result.winning_bidder_id
        self._env = WistEnvironment(rnd.state)
        self._team_tricks = {0: 0, 1: 0}
        self.trick_num = 1

        self._update_hands()
        self.state = "playing"
        self.last_action_time = pygame.time.get_ticks()

    def _play_one_trick(self):
        """Play a single trick."""
        trick_cards, winner, winner_team = play_trick(
            self._round, self._env, self._agents_list,
            self._players, self.discovery, use_mcts=True,
        )

        self._team_tricks[self._players[winner].team_id] += 1
        self.discovery.trick_reward(won=(winner_team == 0))

        self.current_trick_cards = trick_cards
        self.last_winner = winner
        self._update_hands()

        self._log(f"  T{self.trick_num}: {' '.join(f'P{pid}:{c.rank.symbol}{c.suit.symbol}' for pid, c in trick_cards)} -> P{winner}")
        self.trick_num += 1
        self.last_action_time = pygame.time.get_ticks()

        if self.trick_num > 13:
            self._finish_shota()

    def _finish_shota(self):
        """Score the completed shota and check milestones."""
        res = self._tasmiya_result
        tt = self._team_tricks
        scores = score_completed_shota(res, tt)

        self.team_scores[0] += scores.get(0, 0)
        self.team_scores[1] += scores.get(1, 0)
        self.shota_scores.append({0: scores.get(0, 0), 1: scores.get(1, 0)})
        self.discovery.reward(float(scores.get(0, 0)))
        self.shotas_played += 1

        bid_met = tt[res.playing_team_id] >= res.winning_bid_value
        s0 = scores.get(0, 0)

        # Update stats.
        if res.playing_team_id == 0:
            if bid_met:
                self.bids_met += 1
                self._bids_met_t1 += 1
                self._bids_met_streak += 1
            else:
                self.bids_failed += 1
                self._bids_met_streak = 0
            self._bid_history.append(res.winning_bid_value)
        else:
            if bid_met:
                self._bids_met_t2 += 1

        if tt[0] == 13:
            self.seeks_achieved += 1
            self._seeks_t1 += 1
        if tt[1] == 13:
            self._seeks_t2 += 1

        # Track shotas won per team (team that won more tricks in this shota).
        if tt[0] > tt[1]:
            self._shotas_won_t1 += 1
        elif tt[1] > tt[0]:
            self._shotas_won_t2 += 1

        if s0 > self._best_score:
            self._best_score = s0

        self._log(f"  Result: T1={tt[0]} T2={tt[1]} | Score: {scores.get(0, 0):+d}/{scores.get(1, 0):+d}")

        # Milestones.
        self._run_milestone_checks(tt, res.winning_bid_value, res.playing_team_id, bid_met, scores)

        self._last_score = s0
        if self.shota_num == 3:
            self._score_at_shota3 = self.team_scores[0] - self.team_scores[1]

        self.state = "scoring"
        self.last_action_time = pygame.time.get_ticks()

    def _end_game(self):
        """Handle game completion."""
        winner = 0 if self.team_scores[0] > self.team_scores[1] else 1
        self._log(f"  GAME OVER: Team{winner + 1} wins ({self.team_scores[0]}:{self.team_scores[1]})")
        self.state = "game_over"
        self.last_action_time = pygame.time.get_ticks()

        # Record game result.
        game_won = self.team_scores[0] > self.team_scores[1]
        self._wist_win_history.append(game_won)
        if game_won:
            self._team1_wins += 1
        else:
            self._team2_wins += 1

        # Curriculum stage check.
        self._check_stage_transition()

        # Background training.
        if not self._bg_active:
            self._bg_active = True
            threading.Thread(target=self._bg_train, daemon=True).start()

    # ─── Milestone Integration ──────────────────────────────────────────────────

    def _run_milestone_checks(self, team_tricks, bid, playing_team, bid_met, scores):
        """Run both hardcoded and auto-discovery milestone checks."""
        context = {
            "team_tricks": team_tricks,
            "bid": bid,
            "playing_team": playing_team,
            "bid_met": bid_met,
            "scores": scores,
            "team_scores": self.team_scores,
            "shota_num": self.shota_num,
            "game_num": self.game_num,
            "shotas_played": self.shotas_played,
            "seeks_achieved": self.seeks_achieved,
            "bids_met": self.bids_met,
            "bids_failed": self.bids_failed,
            "wist_win_history": self._wist_win_history,
            "last_score": self._last_score,
            "best_score": self._best_score,
            "bids_met_streak": self._bids_met_streak,
            "bid_history": self._bid_history,
            "score_at_shota3": self._score_at_shota3,
            "episodes": self.discovery.episodes_trained,
            "prev_shota_tricks_0": getattr(self, "_prev_shota_tricks_0", 0),
            "defense_streak": getattr(self, "_defense_streak", 0),
        }

        # Track previous shota tricks for back-to-back seek detection.
        self._prev_shota_tricks_0 = team_tricks[0]

        # Track defense streak (opponents failing their bid consecutively).
        if playing_team == 1 and not bid_met:
            self._defense_streak = getattr(self, "_defense_streak", 0) + 1
        elif playing_team == 1 and bid_met:
            self._defense_streak = 0

        check_milestones(context, self._trigger)

        auto_context = {
            "team_tricks": team_tricks,
            "scores": scores,
            "playing_team": playing_team,
            "bid_met": bid_met,
            "episodes": self.discovery.episodes_trained,
            "wist_win_history": self._wist_win_history,
        }
        auto_discover(self._auto_stats, auto_context, self._trigger)

    def _trigger(self, key, msg):
        """Record a discovered behavior."""
        if key in self._milestones_achieved:
            return

        self._milestones_achieved.add(key)
        self._last_discovery_episode_for_stage = self.discovery.episodes_trained

        # Parse title and description.
        if ":" in msg:
            title = msg.split(":")[0].strip().title()
            base_desc = msg.split(":", 1)[1].strip()
        else:
            title = key.replace("_", " ").title()
            base_desc = msg

        # Compute time info.
        total_compute = self._accumulated_compute + self._get_compute_time()
        total_str = self._format_time(total_compute)

        if self._last_discovery_compute is not None:
            delta_str = self._format_time(total_compute - self._last_discovery_compute)
        else:
            delta_str = "—"

        self._last_discovery_compute = total_compute

        # Build stats.
        total_bids = self.bids_met + self.bids_failed
        bid_accuracy = (self.bids_met / max(total_bids, 1)) * 100
        win_rate = 0
        if self._wist_win_history:
            win_rate = sum(self._wist_win_history) / len(self._wist_win_history) * 100
        # Use background agent episodes if available (more current than main).
        episodes = self.discovery.episodes_trained
        if self._bg_agent:
            episodes = max(episodes, self._bg_agent.episodes_trained)

        # Format: each stat on its own line, blank line, then milestone text.
        if episodes >= 1000000:
            ep_str = f"{episodes / 1000000:.1f}M"
        elif episodes >= 1000:
            ep_str = f"{episodes // 1000}K"
        else:
            ep_str = str(episodes)

        desc = (
            f"{base_desc}\n"
            f"\n"
            f"M-Shota: {episodes}\n"
            f"T-Shotas: {ep_str}\n"
            f"Bid accuracy: {bid_accuracy:.0f}%\n"
            f"Win rate: {win_rate:.0f}%\n"
            f"Compute: {total_str}\n"
            f"Since last: {delta_str}"
        )

        self._milestones_list.append((f"{title}", desc))
        self._log(f"  ** DISCOVERED: {title} **")

        # If user is scrolled down, bump scroll so new item at top doesn't shift view.
        if self._disc_scroll_offset > 0:
            self._disc_scroll_offset += 1

    # ─── Curriculum ─────────────────────────────────────────────────────────────

    def _check_stage_transition(self):
        """Graduate to next opponent curriculum stage (1-15) if conditions are met."""
        if self._opponent_stage >= 15:
            # Stage 15 is infinite — just update best snapshot.
            if len(self._wist_win_history) >= 10:
                if sum(self._wist_win_history[-10:]) / 10 >= 0.7:
                    self._best_snapshot = snapshot_brain(self.discovery)
            return

        episodes = self.discovery.episodes_trained
        stagnated = (episodes - self._last_discovery_episode_for_stage) > STAGE_STAGNATION_THRESHOLD

        config = STAGE_CONFIG.get(self._opponent_stage, {})
        required_wr = config.get("win_rate", 0.7)
        window = config.get("window", 50)

        if len(self._wist_win_history) >= window:
            recent_wr = sum(self._wist_win_history[-window:]) / window
            if recent_wr >= required_wr or stagnated:
                self._opponent_stage += 1
                self._best_snapshot = snapshot_brain(self.discovery)
                if not self._frozen_snapshot:
                    self._frozen_snapshot = self._best_snapshot
                self._last_discovery_episode_for_stage = episodes
                next_config = STAGE_CONFIG.get(self._opponent_stage, {})
                stage_name = next_config.get("name", f"Stage {self._opponent_stage}")
                self._log(f"  ** CURRICULUM: Stage {self._opponent_stage} — {stage_name} (episode {episodes}) **")
                reason = f"{recent_wr*100:.0f}% win rate" if not stagnated else "Stagnation detected"
                self._trigger(f"stage_{self._opponent_stage}",
                              f"CURRICULUM STAGE {self._opponent_stage} ({stage_name}): {reason}.")
        elif stagnated:
            self._opponent_stage += 1
            self._best_snapshot = snapshot_brain(self.discovery)
            if not self._frozen_snapshot:
                self._frozen_snapshot = self._best_snapshot
            self._last_discovery_episode_for_stage = episodes
            next_config = STAGE_CONFIG.get(self._opponent_stage, {})
            stage_name = next_config.get("name", f"Stage {self._opponent_stage}")
            self._log(f"  ** CURRICULUM: Stage {self._opponent_stage} — {stage_name} (episode {episodes}) **")
            self._trigger(f"stage_{self._opponent_stage}",
                          f"CURRICULUM STAGE {self._opponent_stage} ({stage_name}): Stagnation detected.")

    # ─── Background Training ────────────────────────────────────────────────────

    def _bg_train(self):
        """Run continuous background self-play training."""
        while self.running:
            # Wait while paused.
            while self.paused and self.running:
                import time as _time
                _time.sleep(0.1)

            if not self.running:
                break

            agent = create_training_clone(self.discovery)
            opp = create_opponent(self.discovery, self._opponent_stage,
                                  self._frozen_snapshot, self._best_snapshot)
            self._bg_agent = agent

            # Bind agent to avoid closure issues across loop iterations.
            _agent = agent

            def milestone_cb(tt, bid, playing_team, bid_met, scores, _a=_agent):
                context = {
                    "team_tricks": tt, "scores": scores,
                    "playing_team": playing_team, "bid_met": bid_met,
                    "episodes": _a.episodes_trained,
                    "wist_win_history": self._wist_win_history,
                }
                auto_discover(self._auto_stats, context, self._trigger)

            win_history = run_background_training(agent, opp, num_shotas=10000,
                                                  milestone_callback=milestone_cb,
                                                  should_pause=lambda: self.paused)
            self._wist_win_history.extend(win_history)

            # Cap win history to avoid unbounded memory growth.
            if len(self._wist_win_history) > 5000:
                self._wist_win_history = self._wist_win_history[-5000:]

            # Sync back.
            self.discovery.episodes_trained = agent.episodes_trained
            self.discovery.total_updates = agent.total_updates
            self.discovery.epsilon = agent.epsilon

            # Auto-save every batch to prevent data loss on crash.
            try:
                self.discovery.save(self.model_path)
                save_milestones(self._milestones_achieved, self._milestones_list,
                                self._accumulated_compute + self._get_compute_time(),
                                self._get_session_stats())
            except Exception:
                pass

        self._bg_active = False
        self._bg_agent = None

    # ─── Insights ───────────────────────────────────────────────────────────────

    def _refresh_insights(self):
        """Refresh cached insights if enough episodes have passed."""
        current_ep = self.discovery.episodes_trained
        if self._bg_agent:
            current_ep = max(current_ep, self._bg_agent.episodes_trained)
        if current_ep - self._last_insight_episode >= 2000 or not self._cached_insights:
            old_count = len(self._cached_insights)
            self._cached_insights = generate_insights(self.discovery)
            self._last_insight_episode = current_ep
            # If user is scrolled and new insights appeared, bump scroll to stay stable.
            new_count = len(self._cached_insights)
            if self._insight_scroll_offset > 0 and new_count > old_count:
                self._insight_scroll_offset += (new_count - old_count)

    # ─── Helpers ────────────────────────────────────────────────────────────────

    def _sync_opponent(self):
        """Sync opponent Q-tables with discovery agent."""
        self.opp.play_q = self.discovery.play_q
        self.opp.play_q2 = self.discovery.play_q2
        self.opp.bid_q = self.discovery.bid_q
        self.opp.bid_q2 = self.discovery.bid_q2
        self.opp.epsilon = self.discovery.epsilon

    def _update_hands(self):
        """Update displayed hands from player objects."""
        self._hands = {p.player_id: sort_hand(p.hand) for p in self._players}

    def _get_compute_time(self) -> float:
        """Get actual compute seconds (excludes paused time)."""
        now = time.monotonic()
        total_elapsed = now - self._app_start_time
        paused = self._total_paused_time
        if self._pause_start is not None:
            paused += now - self._pause_start
        return total_elapsed - paused

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into human-readable short form."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        return f"{seconds / 3600:.1f}h"

    def _get_session_stats(self) -> dict:
        """Build session stats dict for persistence."""
        return {
            "team1_wins": self._team1_wins,
            "team2_wins": self._team2_wins,
            "bids_met_t1": self._bids_met_t1,
            "bids_met_t2": self._bids_met_t2,
            "seeks_t1": self._seeks_t1,
            "seeks_t2": self._seeks_t2,
            "opponent_stage": self._opponent_stage,
            "daks_t1": self._daks_t1,
            "daks_t2": self._daks_t2,
            "shotas_won_t1": self._shotas_won_t1,
            "shotas_won_t2": self._shotas_won_t2,
            "shotas_played": self.shotas_played,
            "seeks_achieved": self.seeks_achieved,
            "bids_met": self.bids_met,
            "bids_failed": self.bids_failed,
            "bids_met_streak": self._bids_met_streak,
            "best_score": self._best_score,
        }

    # ─── Main Loop ──────────────────────────────────────────────────────────────

    def run(self):
        while self.running:
            self._handle_events()
            self._update()
            self._render()
            self.clock.tick(FPS)

        self.discovery.save(self.model_path)
        save_milestones(self._milestones_achieved, self._milestones_list,
                        self._accumulated_compute + self._get_compute_time(),
                        self._get_session_stats())
        pygame.quit()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self._toggle_pause()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # If info overlay is showing, any click closes it.
                if self._show_info:
                    self._show_info = False
                    continue
                if self._mode_btn_rect and self._mode_btn_rect.collidepoint(event.pos):
                    self._toggle_pause()
                if self._reset_btn_rect and self._reset_btn_rect.collidepoint(event.pos):
                    self._reset_brain()
                # Check info label click.
                info_rect = getattr(self, '_info_rect', None)
                if info_rect and info_rect.collidepoint(event.pos):
                    self._show_info = True
                # Check insight category filter chip clicks.
                for cat, rect in self._insight_chip_rects.items():
                    if rect.collidepoint(event.pos):
                        if self._insight_filter == cat:
                            self._insight_filter = None  # Toggle off.
                        else:
                            self._insight_filter = cat  # Toggle on.
                        self._insight_scroll_offset = 0
                        break
                # Check confidence filter clicks (range chips).
                for key, rect in getattr(self, '_conf_rects', {}).items():
                    if rect.collidepoint(event.pos):
                        if self._confidence_filter == key:
                            self._confidence_filter = None  # Toggle off.
                        else:
                            self._confidence_filter = key  # Toggle on (key = low end of range).
                        self._insight_scroll_offset = 0
                        break
            elif event.type == pygame.MOUSEWHEEL:
                mx, _my = pygame.mouse.get_pos()
                right_panel_x = SCREEN_WIDTH - 290
                if mx >= right_panel_x:
                    self._disc_scroll_offset = max(0, self._disc_scroll_offset - event.y)
                elif mx <= 260:
                    self._insight_scroll_offset = max(0, self._insight_scroll_offset - event.y)

    def _toggle_pause(self):
        """Toggle pause and track paused time accurately."""
        if self.paused:
            if self._pause_start is not None:
                self._total_paused_time += time.monotonic() - self._pause_start
                self._pause_start = None
        else:
            self._pause_start = time.monotonic()
        self.paused = not self.paused

    def _reset_brain(self):
        """Reset discovery agent — start from scratch."""
        self.discovery = WistDiscoveryAgent(training=True)
        self._sync_opponent()

        self.shotas_played = 0
        self.seeks_achieved = 0
        self.bids_met = 0
        self.bids_failed = 0
        self._milestones_achieved.clear()
        self._milestones_list.clear()
        self._cached_insights = []
        self._last_insight_episode = 0
        self._auto_stats = create_auto_stats()
        self._opponent_stage = 1
        self._frozen_snapshot = None
        self._best_snapshot = None
        self._last_discovery_episode_for_stage = 0
        self._wist_win_history.clear()
        self._bid_history.clear()
        self._bids_met_streak = 0
        self._last_score = None
        self._best_score = -999
        self._team1_wins = 0
        self._team2_wins = 0
        self._bids_met_t1 = 0
        self._bids_met_t2 = 0
        self._seeks_t1 = 0
        self._seeks_t2 = 0
        self._daks_t1 = 0
        self._daks_t2 = 0
        self._shotas_won_t1 = 0
        self._shotas_won_t2 = 0

        self._accumulated_compute = 0.0
        self._app_start_time = time.monotonic()
        self._total_paused_time = 0.0
        self._pause_start = None
        self._last_discovery_compute = None

        save_milestones(self._milestones_achieved, self._milestones_list, 0.0,
                        self._get_session_stats())
        self._log("  BRAIN RESET -- starting from zero.")

    def _update(self):
        if self.paused:
            return
        now = pygame.time.get_ticks()
        delay = int(TRICK_DELAY_MS / max(self.speed, 2.0))
        if now - self.last_action_time < delay:
            return

        if self.state == "playing":
            self._play_one_trick()
        elif self.state == "scoring":
            self._start_new_shota()
        elif self.state == "game_over":
            self._start_new_game()

    def _render(self):
        """Gather state and delegate to renderer."""
        self._refresh_insights()

        bg_agent = self._bg_agent
        episodes = bg_agent.episodes_trained if bg_agent else self.discovery.episodes_trained

        render_state = {
            "paused": self.paused,
            "game_num": self.game_num,
            "shota_num": self.shota_num,
            "speed": self.speed,
            "team_scores": self.team_scores,
            "shota_scores": self.shota_scores,
            "hands": self._hands,
            "current_trick_cards": self.current_trick_cards,
            "last_winner": self.last_winner,
            "compute_time": self._accumulated_compute + self._get_compute_time(),
            "episodes": episodes,
            "seeks_achieved": self.seeks_achieved + self._auto_stats.get("total_seeks", 0),
            "bids_met": self.bids_met,
            "bids_failed": self.bids_failed,
            "team1_wins": self._team1_wins,
            "team2_wins": self._team2_wins,
            "bids_met_t1": self._bids_met_t1,
            "bids_met_t2": self._bids_met_t2,
            "seeks_t1": self._seeks_t1,
            "seeks_t2": self._seeks_t2,
            "daks_t1": self._daks_t1,
            "daks_t2": self._daks_t2,
            "shotas_won_t1": self._shotas_won_t1,
            "shotas_won_t2": self._shotas_won_t2,
            "epsilon": self.discovery.epsilon,
            "opponent_stage": self._opponent_stage,
            "milestones_list": self._milestones_list,
            "insights": self._cached_insights,
            "insight_filter": self._insight_filter,
            "confidence_filter": self._confidence_filter,
            "disc_scroll": self._disc_scroll_offset,
            "insight_scroll": self._insight_scroll_offset,
            "show_info": self._show_info,
        }

        self._mode_btn_rect, self._reset_btn_rect = self.renderer.render_frame(render_state)

        # Capture chip rects and info rect from renderer for click detection.
        self._insight_chip_rects = render_state.get("_chip_rects", {})
        self._info_rect = render_state.get("_info_rect")
        self._conf_rects = render_state.get("_conf_rects", {})


def main():
    app = WistDiscoveryWatcher()
    app.run()


if __name__ == "__main__":
    main()
