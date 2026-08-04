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
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
from gui_wist_discovery.constants import *
from gui_wist.card_renderer import create_card_surface, create_card_back

from agents.wist_discovery.discovery_agent import WistDiscoveryAgent
from environments.wist.environment import WistEnvironment
from environments.wist.round import Round
from environments.wist.rules import trick_winner
from environments.wist.scoring import score_shota
from environments.wist.setup import create_standard_players
from environments.wist.tasmiya_engine import TasmiyaEngine
from environments.wist.trick import Trick
from intelligence.core.cards.suit import Suit
from intelligence.core.cards.rank import Rank

SUIT_ORDER = {Suit.SPADES: 0, Suit.HEARTS: 1, Suit.CLUBS: 2, Suit.DIAMONDS: 3}
SUIT_IDX = SUIT_ORDER
RANK_ORDER = {
    Rank.TWO: 2, Rank.THREE: 3, Rank.FOUR: 4, Rank.FIVE: 5, Rank.SIX: 6,
    Rank.SEVEN: 7, Rank.EIGHT: 8, Rank.NINE: 9, Rank.TEN: 10,
    Rank.JACK: 11, Rank.QUEEN: 12, Rank.KING: 13, Rank.ACE: 14,
}


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
            "medium": pygame.font.SysFont("Segoe UI", 12),
            "small": pygame.font.SysFont("Segoe UI", 10),
        }

        # Mode.
        self.mode = "follow"
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

        # Opponent shares Q-tables.
        self.opp = WistDiscoveryAgent(training=False)
        self.opp.play_q = self.discovery.play_q
        self.opp.bid_q = self.discovery.bid_q
        self.opp.epsilon = self.discovery.epsilon

        # Game state.
        self.shota_num = 0
        self.game_num = 0
        self.team_scores = [0, 0]
        self.trick_num = 0
        self.current_trick_cards = []
        self.last_winner = -1
        self.event_log = []
        self.last_action_time = 0
        self._card_cache = {}
        self._card_back = create_card_back(CARD_WIDTH, CARD_HEIGHT)
        self._continue_btn_rect = None
        self._mode_btn_rect = None
        self._reset_btn_rect = None
        self._log_scroll_offset = 0
        self._log_auto_scroll = True

        # Milestones.
        self._milestones_achieved = set()
        self._milestones_list = []  # Ordered discovered behaviors.
        self._milestone_queue = []
        self._milestone_announcement = None

        # Load previously discovered milestones.
        self._load_milestones()

        # Stats.
        self.shotas_played = 0
        self.seeks_achieved = 0
        self.bids_met = 0
        self.bids_failed = 0

        self._log(f"Wist Discovery Agent: {self.discovery.episodes_trained} shotas learned")
        self._start_new_game()

    def _log(self, msg):
        self.event_log.append(msg)
        if len(self.event_log) > 200:
            self.event_log = self.event_log[-200:]

    def _start_new_game(self):
        self.game_num += 1
        self.shota_num = 0
        self.team_scores = [0, 0]
        self._log(f"{'='*30} Game #{self.game_num} {'='*30}")
        self._start_new_shota()

    def _start_new_shota(self):
        self.shota_num += 1
        if self.shota_num > 5:
            self._end_game()
            return
        self.trick_num = 0
        self.current_trick_cards = []
        self._log(f"--- Shota {self.shota_num}/5 ---")

        # Setup the shota.
        self._players = create_standard_players()
        self._agents = [self.discovery, self.opp, self.discovery, self.opp]
        self._round = Round(self._players)
        self._round.deal()

        if self._round.has_card_based_dak():
            self.discovery.reset_episode()
            self._log("  Dak (re-deal)")
            self.state = "scoring"
            self.last_action_time = pygame.time.get_ticks()
            return

        # Bidding.
        tasmiya = TasmiyaEngine()
        qabool_id = (self.shota_num - 1) % 4
        try:
            self._tasmiya_result = tasmiya.run(
                players=self._players, agents=self._agents, sahib_al_qabool_id=qabool_id)
        except (ValueError, Exception):
            # Invalid bid happened — skip this shota.
            self.discovery.reset_episode()
            self._log("  Bidding error — skipping")
            self.state = "scoring"
            self.last_action_time = pygame.time.get_ticks()
            return

        if self._tasmiya_result.is_dak:
            self.discovery.reset_episode()
            self._log("  Pass Dak")
            self.state = "scoring"
            self.last_action_time = pygame.time.get_ticks()
            return

        self._log(f"  Bid: {self._tasmiya_result.winning_bid_value} by P{self._tasmiya_result.winning_bidder_id}")

        # Setup play.
        self._round.state.trump_suit = self._tasmiya_result.trump_suit
        self._round.state.winning_bidder_id = self._tasmiya_result.winning_bidder_id
        self._round.next_leading_player_id = self._tasmiya_result.winning_bidder_id
        self._env = WistEnvironment(self._round.state)
        self._team_tricks = {0: 0, 1: 0}
        self.trick_num = 1

        # Store hands for display.
        self._hands = {p.player_id: sorted(
            p.hand, key=lambda c: (SUIT_IDX[c.suit], RANK_ORDER[c.rank])
        ) for p in self._players}

        self.state = "playing"
        self.last_action_time = pygame.time.get_ticks()

    def _play_one_trick(self):
        """Play a single trick."""
        r = self._round
        lid = r.next_leading_player_id
        r.state.current_trick = Trick(leading_player_id=lid)
        play_order = [(lid + i) % 4 for i in range(4)]

        trick_cards = []
        for pid in play_order:
            obs = self._env.observe(pid)
            action = self._agents[pid].act(obs)
            self._env.apply_action(action)
            trick_cards.append((pid, action.card))

        completed = r.state.current_trick
        winner = trick_winner(completed, r.state.trump_suit)
        r.state.completed_tricks.append(completed)
        r.state.current_trick = None
        r.next_leading_player_id = winner
        self._team_tricks[self._players[winner].team_id] += 1

        self.current_trick_cards = trick_cards
        self.last_winner = winner

        # Update hands display.
        self._hands = {p.player_id: sorted(
            p.hand, key=lambda c: (SUIT_IDX[c.suit], RANK_ORDER[c.rank])
        ) for p in self._players}

        self._log(f"  T{self.trick_num}: {' '.join(f'P{pid}:{c.rank.symbol}{c.suit.symbol}' for pid,c in trick_cards)} -> P{winner}")
        self.trick_num += 1
        self.last_action_time = pygame.time.get_ticks()

        if self.trick_num > 13:
            self._finish_shota()

    def _finish_shota(self):
        """Score the completed shota."""
        res = self._tasmiya_result
        tt = self._team_tricks
        scores = score_shota(
            playing_team_id=res.playing_team_id,
            defending_team_id=1 - res.playing_team_id,
            bid=res.winning_bid_value,
            playing_team_tricks=tt[res.playing_team_id],
            defending_team_tricks=tt[1 - res.playing_team_id])

        self.team_scores[0] += scores.get(0, 0)
        self.team_scores[1] += scores.get(1, 0)
        self.discovery.reward(float(scores.get(0, 0)))
        self.shotas_played += 1

        bid_met = tt[res.playing_team_id] >= res.winning_bid_value
        if res.playing_team_id == 0:
            if bid_met:
                self.bids_met += 1
            else:
                self.bids_failed += 1
        if tt[0] == 13 or tt[1] == 13:
            self.seeks_achieved += 1

        self._log(f"  Result: T1={tt[0]} T2={tt[1]} | Score: {scores.get(0,0):+d}/{scores.get(1,0):+d}")
        self._check_milestones(tt, res.winning_bid_value, res.playing_team_id, bid_met, scores)

        self.state = "scoring"
        self.last_action_time = pygame.time.get_ticks()

    def _end_game(self):
        winner = 0 if self.team_scores[0] > self.team_scores[1] else 1
        self._log(f"  GAME OVER: Team{winner+1} wins ({self.team_scores[0]}:{self.team_scores[1]})")
        self.state = "game_over"
        self.last_action_time = pygame.time.get_ticks()

        # Background training.
        if not getattr(self, '_bg_active', False):
            self._bg_active = True
            threading.Thread(target=self._bg_train, daemon=True).start()

    def _bg_train(self):
        """500 silent self-play shotas in background."""
        agent = self.discovery
        opp = WistDiscoveryAgent(training=False)
        opp.play_q = agent.play_q
        opp.bid_q = agent.bid_q
        opp.epsilon = agent.epsilon

        for _ in range(500):
            players = create_standard_players()
            agents = [agent, opp, agent, opp]
            r = Round(players); r.deal()
            if r.has_card_based_dak():
                agent.reset_episode(); continue
            tasmiya = TasmiyaEngine()
            try:
                res = tasmiya.run(players=players, agents=agents, sahib_al_qabool_id=0)
            except (ValueError, Exception):
                agent.reset_episode(); continue
            if res.is_dak:
                agent.reset_episode(); continue
            r.state.trump_suit = res.trump_suit
            r.state.winning_bidder_id = res.winning_bidder_id
            r.next_leading_player_id = res.winning_bidder_id
            env = WistEnvironment(r.state)
            tt = {0: 0, 1: 0}
            for _ in range(13):
                lid = r.next_leading_player_id
                r.state.current_trick = Trick(leading_player_id=lid)
                for pid in [(lid + j) % 4 for j in range(4)]:
                    obs = env.observe(pid)
                    action = agents[pid].act(obs)
                    env.apply_action(action)
                trick = r.state.current_trick
                w = trick_winner(trick, r.state.trump_suit)
                r.state.completed_tricks.append(trick)
                r.state.current_trick = None
                r.next_leading_player_id = w
                tt[players[w].team_id] += 1
            scores = score_shota(
                playing_team_id=res.playing_team_id,
                defending_team_id=1 - res.playing_team_id,
                bid=res.winning_bid_value,
                playing_team_tricks=tt[res.playing_team_id],
                defending_team_tricks=tt[1 - res.playing_team_id])
            agent.reward(float(scores[0]))
            if agent.episodes_trained % 100 == 0 and agent.epsilon > 0.05:
                agent.epsilon *= 0.995
        self._bg_active = False

    def _check_milestones(self, team_tricks, bid, playing_team, bid_met, scores):
        """Detect Wist behavioral milestones — positive discoveries only."""
        t0 = team_tricks[0]  # AI team tricks.
        t1 = team_tricks[1]  # Opponent tricks.
        s0 = scores.get(0, 0)

        # === BASIC (first games) ===
        self._trigger("first_shota", "FIRST SHOTA: Played all 13 tricks successfully.")

        if t0 > 0:
            self._trigger("first_trick", "FIRST TRICK WON: The AI's team won at least one trick.")

        if s0 >= 0:
            self._trigger("no_loss", "ZERO LOSS SHOTA: Scored 0 or positive -- no penalty.")

        if s0 > 0:
            self._trigger("positive_score", "POSITIVE SCORE: Earned points this shota.")

        if t0 > t1:
            self._trigger("won_majority", "WON MAJORITY: Won more tricks than the opponent.")

        # === BIDDING ===
        if playing_team == 0 and bid_met:
            self._trigger("bid_met", "BID MET: Bid and delivered the promised tricks.")

        if playing_team == 0 and bid_met and bid == t0:
            self._trigger("efficient_win", "EFFICIENT WIN: Met bid with exactly the right number of tricks.")

        if playing_team == 0 and bid <= 8 and bid_met and t0 > bid:
            self._trigger("conservative_bid", "CONSERVATIVE BID: Bid low and exceeded it -- under-promise, over-deliver.")

        if playing_team == 0 and bid >= 10 and bid_met:
            self._trigger("aggressive_bid", "AGGRESSIVE BID: Bid 10+ and met it -- confidence and accuracy.")

        if playing_team == 1 and not bid_met:
            self._trigger("defended_well", "DEFENDED WELL: Opponent bid and failed -- the AI's team stopped them.")

        # === TRICK-LEVEL ===
        if t0 >= 3:
            self._trigger("trick_streak", "TRICK STREAK: Won 3+ tricks -- building momentum.")

        if t0 >= 11:
            self._trigger("near_seek", "NEAR SEEK: Won 11+ tricks -- almost got the seek.")

        if t0 == 13:
            self._trigger("seek", "SEEK: Won ALL 13 tricks -- instant game win territory.")

        # === SHOTA ===
        if hasattr(self, '_last_score') and s0 > self._last_score:
            self._trigger("improved", "IMPROVED: Scored better than the previous shota.")
        self._last_score = s0

        if not hasattr(self, '_best_score'):
            self._best_score = s0
        if s0 > self._best_score:
            self._best_score = s0
            if s0 > 5:
                self._trigger("best_yet", "BEST SHOTA YET: Highest score in any shota so far.")

        # === GAME ===
        if self.team_scores[0] >= 25:
            self._trigger("first_win", "FIRST WIN: Reached 25 points and won a game.")

        if self.team_scores[0] >= 25 and self.team_scores[0] - self.team_scores[1] >= 10:
            self._trigger("dominated", "DOMINATED: Won a game by 10+ point margin.")

        if self.shotas_played >= 3 and s0 > 0:
            self._trigger("consistent", "CONSISTENCY: Multiple shotas with positive scores -- strategy forming.")

        # Track win history for game-level milestones.
        if not hasattr(self, '_wist_win_history'):
            self._wist_win_history = []
        if self.shota_num == 5:
            game_won = self.team_scores[0] > self.team_scores[1]
            self._wist_win_history.append(game_won)
            if len(self._wist_win_history) >= 3 and all(self._wist_win_history[-3:]):
                self._trigger("win_streak", "WIN STREAK: Won 3 games in a row.")
            if len(self._wist_win_history) >= 10:
                recent = sum(self._wist_win_history[-10:])
                if recent >= 6:
                    self._trigger("mastery", "MASTERY: Win rate above 60% over 10 games.")
            if len(self._wist_win_history) >= 10:
                early = sum(self._wist_win_history[:5]) / 5
                late = sum(self._wist_win_history[-5:]) / 5
                if late - early >= 0.2:
                    self._trigger("learning_curve", "LEARNING CURVE: Win rate jumped 20%+ from early to recent games.")

        # === PARTNER COOPERATION ===

        # Partner support — partner team won bid and AI contributed tricks.
        if playing_team == 0 and bid_met and t0 >= bid:
            if t0 - bid <= 2:
                self._trigger("team_delivery", "TEAM DELIVERY: The team met their bid with tight execution -- both partners contributed.")

        # Defensive partnership — both partners held opponents below their bid.
        if playing_team == 1 and not bid_met:
            opp_deficit = bid - t1
            if opp_deficit >= 3:
                self._trigger("defensive_wall", "DEFENSIVE WALL: Held opponents 3+ tricks below their bid -- strong partnership defense.")

        # Partner relay — AI's team won tricks alternately (teamwork rhythm).
        if hasattr(self, '_trick_winners') and len(getattr(self, '_trick_winners', [])) >= 6:
            winners = self._trick_winners
            team_wins = [1 if self._players[w].team_id == 0 else 0 for w in winners]
            alternating = sum(1 for i in range(1, len(team_wins))
                             if team_wins[i] != team_wins[i-1])
            if alternating >= 8 and t0 >= 7:
                self._trigger("partner_relay", "PARTNER RELAY: Team won tricks in alternating rhythm -- coordinated play.")

        # === OPPONENT EXPLOITATION ===

        # Over-bid punishment — opponents bid high and AI's team stopped them hard.
        if playing_team == 1 and bid >= 9 and not bid_met:
            self._trigger("overbid_punish", "OVER-BID PUNISHMENT: Opponents bid 9+ and failed -- exploited their overconfidence.")

        # Trump dominance — AI used trump to win multiple tricks.
        if hasattr(self, '_trump_wins'):
            trump_wins = getattr(self, '_trump_wins', 0)
            if trump_wins >= 4:
                self._trigger("trump_dominance", "TRUMP DOMINANCE: Won 4+ tricks using trump -- controlled the game through trump power.")

        # Opponent starved — opponents won 3 or fewer tricks total.
        if t1 <= 3:
            self._trigger("opponent_starved", "OPPONENT STARVED: Opponents won 3 or fewer tricks total -- complete tactical suffocation.")

        # Cut opponent's winner — trumped an opponent's high card in a side suit.
        # (Tracked indirectly by low opponent tricks when they bid high)
        if playing_team == 1 and bid >= 8 and t1 <= bid - 4:
            self._trigger("opponent_crushed", "OPPONENT CRUSHED: Opponents missed their bid by 4+ tricks -- devastating defensive play.")

        # === GAME-WINNING STRATEGIES ===

        # Seek victory — won all 13 tricks and the massive bonus.
        if t0 == 13 and s0 > 20:
            self._trigger("seek_victory", "SEEK VICTORY: Won ALL 13 tricks with a huge score bonus -- the ultimate Wist achievement.")

        # Perfect bid — bid exactly what was won, no waste.
        if playing_team == 0 and bid_met and bid == t0 and bid >= 7:
            self._trigger("perfect_bid", "PERFECT BID: Bid 7+ and won exactly that many tricks -- precision hand evaluation.")

        # Score explosion — earned 15+ points in a single shota.
        if s0 >= 15:
            self._trigger("score_explosion", "SCORE EXPLOSION: Earned 15+ points in a single shota -- maximized scoring opportunity.")

        # Comeback victory — won the game after being behind at shota 3.
        if self.shota_num == 5:
            game_won = self.team_scores[0] > self.team_scores[1]
            if hasattr(self, '_score_at_shota3') and self._score_at_shota3 is not None:
                if self._score_at_shota3 < 0 and game_won:
                    self._trigger("comeback_win", "COMEBACK WIN: Was behind at shota 3 but rallied to win the game -- adaptive strategy.")
                self._score_at_shota3 = None
        if self.shota_num == 3:
            self._score_at_shota3 = self.team_scores[0] - self.team_scores[1]

        # Runaway game — won the game by 20+ points.
        if self.shota_num == 5 and self.team_scores[0] - self.team_scores[1] >= 20:
            self._trigger("runaway_game", "RUNAWAY GAME: Won by 20+ points -- complete strategic superiority over 5 shotas.")

        # === ADVANCED PLAY PATTERNS ===

        # Bid escalation — AI bid higher over multiple games (increasing confidence).
        if not hasattr(self, '_bid_history'):
            self._bid_history = []
        if playing_team == 0:
            self._bid_history.append(bid)
        if len(getattr(self, '_bid_history', [])) >= 5:
            recent_bids = self._bid_history[-5:]
            if all(recent_bids[i] <= recent_bids[i+1] for i in range(len(recent_bids)-1)) and recent_bids[-1] >= 9:
                self._trigger("bid_escalation", "BID ESCALATION: Bids increasing over 5 shotas -- growing confidence in hand evaluation.")

        # Trump conservation — won tricks without using trump, saving trump for later.
        # Detected by: won many tricks overall but opponent still had high cards trumped late.
        if t0 >= 9 and playing_team == 0 and bid_met:
            self._trigger("dominant_play", "DOMINANT PLAY: Won 9+ tricks and met the bid -- overwhelming hand strength exploited perfectly.")

        # Defensive precision — as defending team, won exactly enough to stop the bid.
        if playing_team == 1 and not bid_met and t0 == (14 - bid):
            self._trigger("defensive_precision", "DEFENSIVE PRECISION: Won exactly enough tricks to stop the opponent's bid -- surgical defense.")

        # Consistent bidder — met bid in 3+ consecutive shotas.
        if not hasattr(self, '_bids_met_streak'):
            self._bids_met_streak = 0
        if playing_team == 0 and bid_met:
            self._bids_met_streak += 1
        elif playing_team == 0 and not bid_met:
            self._bids_met_streak = 0
        if getattr(self, '_bids_met_streak', 0) >= 3:
            self._trigger("reliable_bidder", "RELIABLE BIDDER: Met bid in 3+ consecutive shotas -- accurate hand assessment.")

        # High-low strategy — bid low in one shota (and exceeded), bid high in next (and met).
        if len(getattr(self, '_bid_history', [])) >= 2 and playing_team == 0:
            prev_bid = self._bid_history[-2] if len(self._bid_history) >= 2 else 0
            if prev_bid <= 7 and bid >= 10 and bid_met:
                self._trigger("high_low_strategy", "HIGH-LOW STRATEGY: Alternated between conservative and aggressive bids successfully -- adaptive bidding.")

        # === META-MASTERY ===

        # Unbeatable streak — won 5 games in a row.
        if len(self._wist_win_history) >= 5 and all(self._wist_win_history[-5:]):
            self._trigger("unbeatable", "UNBEATABLE: Won 5 games in a row -- opponents cannot counter the AI's strategy.")

        # Grand mastery — 70%+ win rate over 20 games.
        if len(self._wist_win_history) >= 20:
            rate = sum(self._wist_win_history[-20:]) / 20
            if rate >= 0.7:
                self._trigger("grand_mastery", "GRAND MASTERY: Win rate exceeds 70% over 20 games -- deep strategic understanding achieved.")

        # Bid accuracy mastery — met bid in 80%+ of bidding shotas over 10 attempts.
        total_bids = self.bids_met + self.bids_failed
        if total_bids >= 10:
            accuracy = self.bids_met / total_bids
            if accuracy >= 0.8:
                self._trigger("bid_mastery", "BID MASTERY: Met bid in 80%+ of attempts over 10+ shotas -- expert hand evaluation.")

        # Score accumulator — total team score exceeds 50 in a single game.
        if self.team_scores[0] >= 50:
            self._trigger("score_accumulator", "SCORE ACCUMULATOR: Team score exceeded 50 in a single game -- sustained excellence across all shotas.")

        # Seek hunter — achieved 3+ seeks across all games.
        if self.seeks_achieved >= 3:
            self._trigger("seek_hunter", "SEEK HUNTER: Achieved 3+ seeks total -- the AI actively pursues the all-13-tricks strategy when possible.")

    def _trigger(self, key, msg):
        """Record a discovered behavior with title and dynamic description.
        
        Descriptions are enriched with live game stats so they reflect
        actual performance rather than static text.
        """
        if key not in self._milestones_achieved:
            self._milestones_achieved.add(key)

            # Build dynamic context from current game state.
            stats = self._build_stats_context()

            if ":" in msg:
                title = msg.split(":")[0].strip()
                base_desc = msg.split(":", 1)[1].strip()
            else:
                title = key.upper()
                base_desc = msg

            # Enrich with actual numbers.
            desc = self._enrich_description(key, base_desc, stats)

            self._milestones_list.append((title, desc))
            self._log(f"  ** DISCOVERED: {title} **")

    def _build_stats_context(self) -> dict:
        """Gather current performance stats for dynamic descriptions."""
        total_bids = self.bids_met + self.bids_failed
        bid_accuracy = (self.bids_met / max(total_bids, 1)) * 100
        win_rate = 0
        if hasattr(self, '_wist_win_history') and len(self._wist_win_history) > 0:
            win_rate = sum(self._wist_win_history) / len(self._wist_win_history) * 100

        q_states = len(self.discovery.play_q) + len(self.discovery.bid_q)
        episodes = self.discovery.episodes_trained
        epsilon = self.discovery.epsilon

        return {
            "games_played": self.game_num,
            "shotas_played": self.shotas_played,
            "win_rate": win_rate,
            "bid_accuracy": bid_accuracy,
            "seeks": self.seeks_achieved,
            "episodes": episodes,
            "q_states": q_states,
            "epsilon": epsilon,
            "team_score": self.team_scores[0],
        }

    def _enrich_description(self, key: str, base_desc: str, stats: dict) -> str:
        """Add live stats to the description based on milestone type."""
        suffix_parts = []

        if stats["episodes"] > 0:
            suffix_parts.append(f"After {stats['episodes']} shotas learned")

        if stats["win_rate"] > 0:
            suffix_parts.append(f"win rate: {stats['win_rate']:.0f}%")

        if stats["bid_accuracy"] > 0 and stats["shotas_played"] >= 3:
            suffix_parts.append(f"bid accuracy: {stats['bid_accuracy']:.0f}%")

        if stats["q_states"] > 100:
            suffix_parts.append(f"{stats['q_states']:,} states explored")

        suffix = f" [{', '.join(suffix_parts)}]" if suffix_parts else ""
        return f"{base_desc}{suffix}"

    def _show_next_milestone(self):
        """No longer used."""
        return False

    def _render_hand_h(self, hand, cx, y, table):
        """Render horizontal hand with fixed 13-card spacing."""
        if not hand:
            return
        overlap = min(CARD_WIDTH - 8, (table.width - 200) // 13)
        start_x = cx - (overlap * 12 + CARD_WIDTH) // 2
        for i, card in enumerate(hand):
            key = f"{card.rank.symbol}{card.suit.symbol}"
            if key not in self._card_cache:
                self._card_cache[key] = create_card_surface(card.rank.symbol, card.suit.symbol, CARD_WIDTH, CARD_HEIGHT)
            self.screen.blit(self._card_cache[key], (start_x + i * overlap, y))

    def _render_hand_v(self, hand, x, start_y):
        """Render vertical hand with fixed 16px overlap."""
        if not hand:
            return
        for i, card in enumerate(hand):
            key = f"{card.rank.symbol}{card.suit.symbol}"
            if key not in self._card_cache:
                self._card_cache[key] = create_card_surface(card.rank.symbol, card.suit.symbol, CARD_WIDTH, CARD_HEIGHT)
            surf = pygame.transform.smoothscale(self._card_cache[key], (CARD_MINI_W, CARD_MINI_H))
            self.screen.blit(surf, (x, start_y + i * 16))

    # === Main loop ===

    def run(self):
        while self.running:
            self._handle_events()
            self._update()
            self._render()
            self.clock.tick(FPS)
        self.discovery.save(self.model_path)
        self._save_milestones()
        pygame.quit()

    def _save_milestones(self):
        """Save discovered milestones to disk."""
        import json
        path = "agents/wist_discovery/milestones.json"
        try:
            data = {"achieved": list(self._milestones_achieved), "list": self._milestones_list}
            with open(path, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def _load_milestones(self):
        """Load previously discovered milestones."""
        import json
        path = "agents/wist_discovery/milestones.json"
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self._milestones_achieved = set(data.get("achieved", []))
            self._milestones_list = [tuple(x) if isinstance(x, list) else x
                                     for x in data.get("list", [])]
        except (FileNotFoundError, Exception):
            pass

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self._continue_btn_rect and self._continue_btn_rect.collidepoint(event.pos):
                    self._on_continue()
                if self._mode_btn_rect and self._mode_btn_rect.collidepoint(event.pos):
                    self.mode = "follow" if self.mode == "watch" else "watch"
                if self._reset_btn_rect and self._reset_btn_rect.collidepoint(event.pos):
                    self._reset_brain()
            elif event.type == pygame.MOUSEWHEEL:
                self._disc_scroll_offset = getattr(self, '_disc_scroll_offset', 0)
                self._disc_scroll_offset -= event.y * 1
                self._disc_scroll_offset = max(0, self._disc_scroll_offset)

    def _on_continue(self):
        if self.state == "scoring":
            self._start_new_shota()
        elif self.state == "game_over":
            self._start_new_game()

    def _reset_brain(self):
        """Reset discovery agent — start from scratch."""
        self.discovery = WistDiscoveryAgent(training=True)
        self.opp.play_q = self.discovery.play_q
        self.opp.bid_q = self.discovery.bid_q
        self.shotas_played = 0
        self.seeks_achieved = 0
        self.bids_met = 0
        self.bids_failed = 0
        self._milestones_achieved.clear()
        self._milestones_list.clear()
        self._save_milestones()
        self._log("  BRAIN RESET -- starting from zero.")

    def _update(self):
        if self.paused:
            return
        now = pygame.time.get_ticks()

        if self.mode == "watch":
            delay = int(TRICK_DELAY_MS / max(self.speed, 2.0))
            if now - self.last_action_time < delay:
                return
            if self.state == "playing":
                self._play_one_trick()
            elif self.state == "scoring":
                self._start_new_shota()
            elif self.state == "game_over":
                self._start_new_game()
        else:
            # Follow mode — auto-advance tricks, wait for Continue on scoring.
            delay = int(TRICK_DELAY_MS * 1.5 / self.speed)
            if now - self.last_action_time < delay:
                return
            if self.state == "playing":
                self._play_one_trick()

    # === Rendering (identical layout to Hearts) ===

    def _render(self):
        self.screen.fill(BG_DARK)

        # Table area.
        table = pygame.Rect(20, 60, SCREEN_WIDTH - 320, SCREEN_HEIGHT - 80)
        pygame.draw.rect(self.screen, TABLE_FELT, table, border_radius=12)
        pygame.draw.rect(self.screen, TABLE_BORDER, table, width=2, border_radius=12)

        # Title.
        t = self.fonts["title"].render(TITLE, True, TEXT_WHITE)
        self.screen.blit(t, (25, 15))

        # Info — right-aligned above panel, two lines, white.
        panel_x = SCREEN_WIDTH - 290
        info1 = f"Game {self.game_num} | Shota {self.shota_num}/5 | Speed: {self.speed:.1f}x"
        info2 = f"Score: T1={self.team_scores[0]:+d} T2={self.team_scores[1]:+d} | SPACE: pause | ESC: quit"
        self.screen.blit(self.fonts["small"].render(info1, True, TEXT_WHITE),
            (panel_x - self.fonts["small"].size(info1)[0] - 10, 15))
        self.screen.blit(self.fonts["small"].render(info2, True, TEXT_WHITE),
            (panel_x - self.fonts["small"].size(info2)[0] - 10, 32))

        if self.paused:
            ps = self.fonts["large"].render("PAUSED", True, TEXT_GOLD)
            self.screen.blit(ps, (SCREEN_WIDTH // 2 - 40, 15))

        # Render players.
        self._render_players(table)

        # Trick in center.
        self._render_trick(table)

        # Continue button (follow mode only).
        self._continue_btn_rect = None
        if self.mode == "follow" and self.state in ("scoring", "game_over"):
            self._render_continue_btn(table)

        # Mode button (top-right of table).
        self._render_mode_btn(table)

        # Reset button (top-left of table).
        self._render_reset_btn(table)

        # Right panel.
        self._render_panel()

        pygame.display.flip()

    def _render_players(self, table):
        """Render 4 players — same layout as Hearts."""
        cx, cy = table.centerx, table.centery
        hands = getattr(self, '_hands', {})

        for pid in range(4):
            hand = hands.get(pid, [])
            color = PLAYER_COLORS[pid]
            name_surf = self.fonts["medium"].render(PLAYER_NAMES[pid], True, color)

            if pid == 0:  # Bottom
                hand_y = table.bottom - CARD_HEIGHT - 50
                self._render_hand_h(hand, cx, hand_y, table)
                self.screen.blit(name_surf, (cx - name_surf.get_width()//2, hand_y + CARD_HEIGHT + 6))
            elif pid == 2:  # Top
                self.screen.blit(name_surf, (cx - name_surf.get_width()//2, table.top + 10))
                self._render_hand_h(hand, cx, table.top + 28, table)
            elif pid == 1:  # Left
                self.screen.blit(name_surf, (table.left + 15, cy - 155))
                self._render_hand_v(hand, table.left + 15, cy - 135)
            elif pid == 3:  # Right
                self.screen.blit(name_surf,
                    (table.right - 15 - name_surf.get_width(), cy - 155))
                self._render_hand_v(hand, table.right - CARD_MINI_W - 15, cy - 135)

    def _render_trick(self, table):
        """Render current trick cards in center."""
        if not self.current_trick_cards:
            return
        cx, cy = table.centerx, table.centery
        offsets = {0: (0, 44), 1: (-70, 0), 2: (0, -44), 3: (70, 0)}
        for pid, card in self.current_trick_cards:
            ox, oy = offsets[pid]
            key = f"{card.rank.symbol}{card.suit.symbol}"
            if key not in self._card_cache:
                self._card_cache[key] = create_card_surface(
                    card.rank.symbol, card.suit.symbol, CARD_WIDTH, CARD_HEIGHT)
            self.screen.blit(self._card_cache[key],
                (cx + ox - CARD_WIDTH//2, cy + oy - CARD_HEIGHT//2))

        if self.last_winner >= 0 and len(self.current_trick_cards) == 4:
            tw = self.fonts["small"].render(f"Taker: {PLAYER_NAMES[self.last_winner]}", True, TEXT_GOLD)
            self.screen.blit(tw, (table.x + 120, table.y + 12))

    def _render_continue_btn(self, table):
        """Continue button — bottom center, same style as Hearts."""
        btn = pygame.Rect(table.centerx - 55, table.bottom - 40, 110, 28)
        self._continue_btn_rect = btn
        hover = btn.collidepoint(pygame.mouse.get_pos())
        bg = (90, 200, 90) if hover else (70, 170, 70)
        pygame.draw.rect(self.screen, bg, btn, border_radius=6)
        pygame.draw.rect(self.screen, (180, 255, 180), btn, width=1, border_radius=6)
        t = self.fonts["medium"].render("Continue", True, (255, 255, 255))
        self.screen.blit(t, t.get_rect(center=btn.center))

    def _render_mode_btn(self, table):
        """Mode toggle — top-right, same as Hearts."""
        btn = pygame.Rect(table.right - 115, table.top + 10, 105, 28)
        self._mode_btn_rect = btn
        hover = btn.collidepoint(pygame.mouse.get_pos())
        label = "Watch Auto" if self.mode == "follow" else "See & Check"
        bg = (30, 100, 180) if not hover else (50, 130, 210)
        if self.mode == "watch":
            bg = (140, 80, 20) if not hover else (180, 110, 40)
        pygame.draw.rect(self.screen, bg, btn, border_radius=6)
        if hover:
            pygame.draw.rect(self.screen, (200, 200, 200), btn, width=1, border_radius=6)
        t = self.fonts["small"].render(label, True, (255, 255, 255))
        self.screen.blit(t, t.get_rect(center=btn.center))
        mode_t = self.fonts["small"].render(
            "Following" if self.mode == "follow" else "Watching", True, TEXT_GOLD)
        self.screen.blit(mode_t, mode_t.get_rect(centerx=btn.centerx, y=btn.bottom + 3))

    def _render_reset_btn(self, table):
        """Reset Brain — top-left, same as Hearts."""
        btn = pygame.Rect(table.left + 10, table.top + 10, 110, 28)
        self._reset_btn_rect = btn
        hover = btn.collidepoint(pygame.mouse.get_pos())
        bg = (140, 40, 40) if hover else (100, 30, 30)
        pygame.draw.rect(self.screen, bg, btn, border_radius=6)
        if hover:
            pygame.draw.rect(self.screen, (200, 200, 200), btn, width=1, border_radius=6)
        t = self.fonts["small"].render("Reset Brain", True, (255, 255, 255))
        self.screen.blit(t, t.get_rect(center=btn.center))

    def _render_panel(self):
        """Render right panel — two boxes: Scoreboard + Discoveries (same as Hearts)."""
        px = SCREEN_WIDTH - 290
        panel_w = 280

        # === Box 1: Scoreboard ===
        score_h = 185
        score_rect = pygame.Rect(px, 60, panel_w, score_h)
        pygame.draw.rect(self.screen, PANEL_DARK, score_rect, border_radius=10)
        pygame.draw.rect(self.screen, (40, 60, 40), score_rect, width=1, border_radius=10)

        y = score_rect.top + 10
        self.screen.blit(self.fonts["large"].render("Stats", True, TEXT_WHITE), (px + 10, y))
        y += 22
        stats = [
            f"Shotas learned: {self.discovery.episodes_trained:,}",
            f"Seeks: {self.seeks_achieved}",
            f"Bids met: {self.bids_met}/{self.bids_met + self.bids_failed}",
            f"Score: T1={self.team_scores[0]:+d}  T2={self.team_scores[1]:+d}",
            f"Q-table: {len(self.discovery.play_q) + len(self.discovery.bid_q)} states",
            f"Epsilon: {self.discovery.epsilon:.3f}",
        ]
        for s in stats:
            self.screen.blit(self.fonts["small"].render(s, True, TEXT_LIGHT), (px + 10, y))
            y += 16

        # === Box 2: Discoveries ===
        disc_top = 60 + score_h + 8
        disc_h = SCREEN_HEIGHT - 80 - score_h - 8
        disc_rect = pygame.Rect(px, disc_top, panel_w, disc_h)
        pygame.draw.rect(self.screen, PANEL_DARK, disc_rect, border_radius=10)
        pygame.draw.rect(self.screen, (70, 60, 20), disc_rect, width=1, border_radius=10)

        self.screen.set_clip(pygame.Rect(px + 5, disc_top + 5, panel_w - 10, disc_h - 10))

        y = disc_rect.top + 10
        self.screen.blit(self.fonts["large"].render("Discoveries", True, TEXT_GOLD), (px + 10, y))
        y += 22

        if not self._milestones_list:
            self.screen.blit(self.fonts["medium"].render("None yet...", True, TEXT_DIM), (px + 15, y))
            self.screen.set_clip(None)
        else:
            disc_scroll = getattr(self, '_disc_scroll_offset', 0)
            line_h = 30
            max_visible = (disc_rect.bottom - y - 15) // line_h
            total = len(self._milestones_list)
            max_scroll = max(0, total - max_visible)
            disc_scroll = max(0, min(disc_scroll, max_scroll))
            self._disc_scroll_offset = disc_scroll

            if disc_scroll == 0:
                visible = self._milestones_list[-max_visible:]
                start_num = max(1, total - max_visible + 1)
            else:
                end_idx = total - disc_scroll
                start_idx = max(0, end_idx - max_visible)
                visible = self._milestones_list[start_idx:end_idx]
                start_num = start_idx + 1

            panel_text_w = panel_w - 30
            for i, (title_text, desc_text) in enumerate(visible):
                num = start_num + i
                is_latest = (disc_scroll == 0 and i == len(visible) - 1)
                title_color = (100, 255, 100) if is_latest else (255, 255, 255)
                self.screen.blit(self.fonts["large"].render(
                    f"{num}. {title_text}", True, title_color), (px + 10, y))
                y += 18

                desc_color = (200, 255, 200) if is_latest else (210, 210, 210)
                desc_font = self.fonts["medium"]
                words = desc_text.split()
                line = ""
                for w in words:
                    test = line + " " + w if line else w
                    if desc_font.size(test)[0] < panel_text_w:
                        line = test
                    else:
                        if y > disc_rect.bottom - 20:
                            break
                        self.screen.blit(desc_font.render(line, True, desc_color), (px + 20, y))
                        y += 15
                        line = w
                if line and y <= disc_rect.bottom - 20:
                    self.screen.blit(desc_font.render(line, True, desc_color), (px + 20, y))
                    y += 17

                if y > disc_rect.bottom - 15:
                    break

        # Reset clip.
        self.screen.set_clip(None)


def main():
    app = WistDiscoveryWatcher()
    app.run()

if __name__ == "__main__":
    main()
