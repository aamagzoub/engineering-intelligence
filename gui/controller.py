
from agents.random.random_agent import RandomAgent
from agents.rule_based.rule_based_agent import RuleBasedAgent
from agents.learning.learning_agent import LearningAgent
from environments.wist.dak import triggers_card_based_dak
from environments.wist.environment import WistEnvironment
from environments.wist.game_state import DakType, GameState
from environments.wist.playing_engine import PlayingEngine
from environments.wist.round import Round
from environments.wist.scoring import detect_seek, score_shota
from environments.wist.setup import create_standard_players
from environments.wist.tasmiya_engine import TasmiyaEngine, determine_trump_suit, determine_first_shota_qabool
from gui.controller_helpers import ControllerHelpersMixin

class SimulationController(ControllerHelpersMixin):
    """
    Coordinates the GUI and the Wist engine.
    Supports:
    - Auto mode: runs from start to end
    - Step mode: pauses after each phase/trick until Continue is pressed
    """

    def __init__(self, app):
        self.app = app
        self.running = False
        self.paused = False
        self.step_mode = False
        self.phase = "idle"

        self.players = None
        self.round = None
        self.environment = None
        self.agents = None
        self.engine = None

        self.current_trick_number = 0
        self.team_tricks = [0, 0]

        self.trump_suit = None
        self.qabool_player_index = None
        self.bid_value = None
        self.player_bids = ["-", "-", "-", "-"]
        self.first_shooter_player_id = 0

        self.shota_setup_engine = TasmiyaEngine()
        self.shota_setup = None

        # Game-level state for multi-Shota tracking.
        self.game_state = GameState()
        self.sahib_al_qabool_id = 0
        self.playing_team_id = None
        self.defending_team_id = None
        self.player_tricks_won = [0, 0, 0, 0]
        self._last_tasmiya_result = None
        self._trick_in_progress = False
        self._auto_game_mode = False
        self._scheduled_timer = None

        # Agent type per player (configurable from GUI).
        # Options: "rule_based", "random", "learning"
        self.agent_types = ["rule_based", "rule_based", "rule_based", "rule_based"]
        self._learning_agent_cache = None  # Shared learning agent instance.

    # ---------------------------------------------------------
    # Command handlers (called by GUI buttons)
    # ---------------------------------------------------------

    def cmd_auto_game(self) -> None:
        """Auto Game: play from first shota to end without stopping."""
        self._cancel_timer()
        self._auto_game_mode = True
        self.step_mode = False
        if not self.running:
            self.start()
        else:
            # If paused, unpause.
            self.paused = False
            self.continue_simulation()

    def cmd_auto_shota(self) -> None:
        """Auto Shota: play one shota automatically, pause between shotas."""
        self._cancel_timer()
        self._auto_game_mode = False
        self.step_mode = False
        if not self.running:
            self.start()
        else:
            self.paused = False
            self.continue_simulation()

    def cmd_continue(self) -> None:
        """Continue: switch to auto-shota mode and resume."""
        self._cancel_timer()
        self._auto_game_mode = False
        self.step_mode = False
        if self.running:
            self.paused = False
            self.continue_simulation()

    def cmd_manual(self) -> None:
        """Manual: switch to step-by-step mode."""
        self._cancel_timer()
        self._auto_game_mode = False
        self.step_mode = True
        if not self.running:
            self.start()
        else:
            self.paused = False

    def cmd_next(self) -> None:
        """Next: advance one step in manual mode."""
        if self.running:
            self._cancel_timer()
            self.step_mode = True
            self._auto_game_mode = False
            self.paused = False
            self.continue_simulation()

    def cmd_pause(self) -> None:
        """Pause the game."""
        self._cancel_timer()
        self.pause()

    def cmd_stop(self) -> None:
        """Stop the game."""
        self._cancel_timer()
        self.stop()

    def cmd_reset(self) -> None:
        """Reset: stop and clear everything for a fresh start."""
        self._cancel_timer()
        self.running = False
        self.paused = False
        self.phase = "idle"
        self._trick_in_progress = False
        self._trick_just_finished = False

        self.game_state = GameState()
        self.sahib_al_qabool_id = 0

        self.app.reset_player_statuses()
        self.app.clear_played_cards()
        self.set_shota_info_safe(trump="-", qabool="-", bid="-", first_shooter="-")
        self.app.set_current_trick("Waiting to start")

        try:
            self.app.shota_info_labels["winner"].config(text="—")
            self.app.shota_counter_label.config(text="1 / 5")
            self.app.trick_counter_label.config(text="— / 13")
            self.app.score_team1_label.config(text="0")
            self.app.score_team2_label.config(text="0")
            self.app.deal_counter_label.config(text="1")
        except (AttributeError, KeyError):
            pass

    # ---------------------------------------------------------
    # Timer management (prevent duplicate scheduled callbacks)
    # ---------------------------------------------------------

    def _schedule(self, delay_ms: int, callback) -> None:
        """Schedule a callback, cancelling any previous one."""
        self._cancel_timer()
        self._scheduled_timer = self.app.root.after(delay_ms, callback)

    def _cancel_timer(self) -> None:
        """Cancel any pending scheduled callback."""
        if self._scheduled_timer is not None:
            try:
                self.app.root.after_cancel(self._scheduled_timer)
            except (ValueError, AttributeError):
                pass
            self._scheduled_timer = None

    # ---------------------------------------------------------
    # Start modes
    # ---------------------------------------------------------

    def start_auto(self) -> None:
        self.step_mode = False
        self.start()

    def start_step_mode(self) -> None:
        self.step_mode = True
        self.start()

    def _make_agent(self, player_index: int):
        """Create an agent based on the configured type for this player."""
        agent_type = self.agent_types[player_index]
        if agent_type == "random":
            return RandomAgent()
        elif agent_type == "learning":
            if self._learning_agent_cache is None:
                self._learning_agent_cache = LearningAgent(training=False)
            return self._learning_agent_cache
        else:
            return RuleBasedAgent()

    def start(self) -> None:
        if self.running:
            self.app.log("Simulation is already running.")
            return

        self.running = True
        self.paused = False
        self.phase = "dealing"

        self.current_trick_number = 0
        self.team_tricks = [0, 0]

        self.trump_suit = None
        self.qabool_player_index = None
        self.bid_value = None
        self.player_bids = ["-", "-", "-", "-"]
        self.first_shooter_player_id = 0
        self.shota_setup = None
        self.playing_team_id = None
        self.defending_team_id = None
        self.player_tricks_won = [0, 0, 0, 0]

        # Reset game state for a new game.
        self.game_state = GameState()
        self.sahib_al_qabool_id = determine_first_shota_qabool()
        self._deal_number = 0  # Will be incremented to 1 in _deal_and_check_dak.
        self._deal_number_counter = 1

        self.app.clear_log()
        self.app.reset_player_statuses()
        self.app.set_status("Simulation running...")
        self.app.set_current_trick("Dealing cards...")
        self.set_shota_info_safe(
            trump="-",
            qabool="-",
            bid="-",
            first_shooter="-",
        )

        self.app.log("Simulation started.")
        self.app.log(f"Mode: {'Step-by-step' if self.step_mode else 'Auto'}")

        self.players = create_standard_players()

        # Apply agent selection from GUI (if available).
        try:
            self.app._apply_agent_selection()
        except (AttributeError, TypeError):
            pass

        self.agents = [
            self._make_agent(0),
            self._make_agent(1),
            self._make_agent(2),
            self._make_agent(3),
        ]

        self.engine = PlayingEngine()

        # Deal and handle card-based Dak.
        self._deal_and_check_dak()

        self.app.log("Players created.")
        self.app.log("Cards dealt.")

        self.show_player_hands()

        self.phase = "dealt"

        # Highlight Sahib Al-Qabool from the very start.
        for i in range(4):
            self.set_player_status_safe(
                i, "Waiting",
                is_qabool=(i == self.sahib_al_qabool_id),
            )

        # Show Qabool in the top bar immediately.
        self.set_shota_info_safe(
            trump="-",
            qabool=f"Player {self.sahib_al_qabool_id + 1}",
            bid="-",
            first_shooter="-",
        )

        # Show current Shota number from the start.
        try:
            current_shota = self.game_state.completed_shotas + 1
            self.app.set_game_score(
                team_1=self.game_state.team_scores[0],
                team_2=self.game_state.team_scores[1],
                shotas=current_shota,
            )
        except (AttributeError, TypeError):
            pass

        if self.step_mode:
            self.app.set_status("Paused after dealing. Press Continue.")
            self.app.set_current_trick(
                "Cards dealt.\nReview all hands, then press Continue."
            )
            return

        self.continue_simulation()

    # ---------------------------------------------------------
    # Phase control
    # ---------------------------------------------------------

    def continue_simulation(self) -> None:
        if not self.running:
            return

        self.paused = False

        if self.phase == "dealt":
            # Show Qabool and pause before bidding starts.
            self.set_shota_info_safe(
                trump="-",
                qabool=f"Player {self.sahib_al_qabool_id + 1}",
                bid="-",
                first_shooter="-",
            )
            for i in range(4):
                self.set_player_status_safe(
                    i, "Waiting",
                    is_qabool=(i == self.sahib_al_qabool_id),
                )
            deal_num = getattr(self, "_deal_number_counter", 1)
            try:
                self.app.set_deal_number(deal_num)
            except (AttributeError, TypeError):
                pass

            self.app.set_current_trick(
                f"Deal {deal_num}\n"
                f"Sahib Al-Qabool: Player {self.sahib_al_qabool_id + 1}\n"
                f"Al-Tasmiya begins..."
            )

            self.phase = "qabool_shown"
            if self.step_mode:
                return
            else:
                self._schedule(700, self.continue_simulation)
                return

        if self.phase == "qabool_shown":
            # Now run the Tasmiya (bidding).
            self._run_one_tasmiya()
            return

        if self.phase == "dak_shown":
            # User acknowledged Dak — clear bids, re-deal and try again.
            for i in range(4):
                self.app.set_player_bid(i, "-")

            self._deal_number_counter = getattr(self, "_deal_number_counter", 1) + 1
            self.sahib_al_qabool_id = (self.sahib_al_qabool_id + 1) % 4
            self._deal_and_check_dak()
            self.show_player_hands()
            self.phase = "dealt"

            # Show new Qabool on top and in table.
            try:
                self.app.set_deal_number(self._deal_number_counter)
            except (AttributeError, TypeError):
                pass
            self.set_shota_info_safe(
                trump="-",
                qabool=f"Player {self.sahib_al_qabool_id + 1}",
                bid="-",
                first_shooter="-",
            )
            for i in range(4):
                self.set_player_status_safe(
                    i, "Waiting",
                    is_qabool=(i == self.sahib_al_qabool_id),
                )

            if self.step_mode:
                self.app.set_current_trick(
                    f"Deal {self._deal_number_counter}\n"
                    f"Qabool: Player {self.sahib_al_qabool_id + 1}\n"
                    f"Review hands, press Next."
                )
                return
            else:
                # Auto: brief pause then continue bidding.
                self._schedule(700, self.continue_simulation)
                return

        if self.phase == "bidding_step":
            self._show_next_bid()
            return

        if self.phase == "setup_ready":
            self.phase = "before_first_trick"
            self.first_shooter_player_id = self.round.next_leading_player_id

            self.set_shota_info_safe(
                trump=self._format_trump(self.trump_suit),
                qabool=f"Player {self.qabool_player_index + 1}",
                bid=str(self.bid_value),
                first_shooter=f"Player {self.first_shooter_player_id + 1}",
            )

            self.app.set_current_trick(
                f"Ready for first trick\n"
                f"Shooter: Player {self.first_shooter_player_id + 1}"
            )

            for player_index in range(4):
                self.set_player_status_safe(
                    player_index,
                    "Ready",
                    is_qabool=(player_index == self.qabool_player_index),
                    is_first_shooter=(player_index == self.first_shooter_player_id),
                )

            if self.step_mode:
                self.app.set_status("Paused before first trick. Press Next.")
                return

        if self.phase == "before_first_trick":
            self.phase = "playing_tricks"

        if self.phase == "playing_tricks":
            if self.step_mode:
                self._play_next_card_step()
            else:
                self._schedule(700, self.play_next_trick_auto)
            return

        if self.phase == "shota_ended":
            # User pressed Next/Continue/Auto after a Shota — start the next one.
            self._reset_for_new_shota()
            if not self.step_mode:
                # Resume auto play for the new Shota (use schedule, not recursion).
                self._schedule(100, self.continue_simulation)
            else:
                self.app.set_current_trick("New Shota dealt.\nReview hands, press Next.")
            return

        if self.phase == "completed":
            return

    # ---------------------------------------------------------
    # Step-by-step bidding
    # ---------------------------------------------------------

    def _run_one_tasmiya(self) -> None:
        """
        Run one round of Tasmiya. If result is Dak, show it and pause.
        If result is a valid bid, proceed to setup_ready.
        On the 3rd deal (pass-based Dak limit reached), force Qabool to play.
        """
        deal_num = getattr(self, "_deal_number_counter", 1)

        # Update GUI with deal number and current Qabool.
        try:
            self.app.set_deal_number(deal_num)
        except (AttributeError, TypeError):
            pass

        # Show Qabool in the top bar immediately.
        self.set_shota_info_safe(
            trump="-",
            qabool=f"Player {self.sahib_al_qabool_id + 1}",
            bid="-",
            first_shooter="-",
        )

        # Mark Qabool player in the table.
        for i in range(4):
            self.set_player_status_safe(
                i, "Waiting",
                is_qabool=(i == self.sahib_al_qabool_id),
            )

        # Check if this is the forced deal (3rd attempt).
        is_forced = (self.game_state.pass_based_dak_count >= 2)

        # Run Tasmiya.
        tasmiya_result = self.shota_setup_engine.run(
            players=self.players,
            agents=self.agents,
            sahib_al_qabool_id=self.sahib_al_qabool_id,
            is_first_shota=(self.game_state.completed_shotas == 0),
        )

        if tasmiya_result.is_dak and not is_forced:
            # Dak declared — in step mode, show bids one by one first.
            self.game_state.apply_dak(DakType.PASS_BASED)
            self._last_tasmiya_result = tasmiya_result
            self._tasmiya_is_dak = True

            if self.step_mode:
                # Show bids step by step, then show Dak at the end.
                self._bid_reveal_index = 0
                self._bid_order = list(tasmiya_result.bid_history)
                for i in range(4):
                    self.app.set_player_bid(i, "-")
                self.phase = "bidding_step"
                self._show_next_bid()
                return
            else:
                # Auto: show all bids then Dak message.
                for player_id, bid_value in tasmiya_result.bid_history:
                    if bid_value is not None:
                        self.app.set_player_bid(player_id, str(bid_value))
                    else:
                        is_q = (player_id == self.sahib_al_qabool_id)
                        self.app.set_player_bid(player_id, "Dak" if is_q else "Pass")

                self.app.set_current_trick(
                    f"دك — DAK #{self.game_state.pass_based_dak_count}\n"
                    f"Deal {deal_num}: All passed\n"
                    f"Sahib Al-Qabool (P{self.sahib_al_qabool_id + 1}) declares Dak"
                )
                self.phase = "dak_shown"
                self._schedule(1200, self.continue_simulation)
                return

        if tasmiya_result.is_dak and is_forced:
            # 3rd Dak — force Qabool to play based on hand strength.
            from collections import Counter
            from environments.wist.tasmiya_engine import tasmiya_order

            qabool_hand = self.players[self.sahib_al_qabool_id].hand
            trump = determine_trump_suit(qabool_hand)
            suit_counts = Counter(c.suit for c in qabool_hand)
            longest_count = max(suit_counts.values())
            forced_bid = max(7, min(longest_count + 3, 13))

            self.qabool_player_index = self.sahib_al_qabool_id
            self.bid_value = forced_bid
            self.trump_suit = trump
            self.playing_team_id = self.players[self.sahib_al_qabool_id].team_id
            self.defending_team_id = 1 if self.playing_team_id == 0 else 0
            self.player_bids = ["Pass", "Pass", "Pass", "Pass"]
            self.player_bids[self.sahib_al_qabool_id] = f"{forced_bid} (forced)"

            self.round.state.trump_suit = self.trump_suit
            self.round.state.winning_bidder_id = self.sahib_al_qabool_id
            self.first_shooter_player_id = self.sahib_al_qabool_id
            self.round.next_leading_player_id = self.sahib_al_qabool_id

            self._last_tasmiya_result = tasmiya_result
            self._tasmiya_is_dak = False  # Not Dak — forced play.

            if self.step_mode:
                # Build synthetic bid history: 3 passes + forced bid.
                order = tasmiya_order(self.sahib_al_qabool_id)
                self._bid_order = [(pid, None) for pid in order]
                self._bid_order.append((self.sahib_al_qabool_id, forced_bid))
                self._bid_reveal_index = 0
                for i in range(4):
                    self.app.set_player_bid(i, "-")
                self.phase = "bidding_step"
                self._show_next_bid()
                return
            else:
                # Auto: show all bids at once.
                for i, bid_text in enumerate(self.player_bids):
                    self.app.set_player_bid(i, bid_text)
                self._update_gui_after_tasmiya()
                self.app.set_current_trick(
                    f"Deal {deal_num}: DAK limit!\n"
                    f"P{self.sahib_al_qabool_id + 1} forced to play\n"
                    f"Bid: {forced_bid} | Trump: {trump.name}"
                )
                self.phase = "setup_ready"
                self._schedule(700, self.continue_simulation)
                return

        # Normal bid — someone bid and Qabool accepted or outbid.
        self._tasmiya_is_dak = False
        self.qabool_player_index = tasmiya_result.sahib_al_qabool_id
        self.bid_value = tasmiya_result.winning_bid_value
        self.trump_suit = tasmiya_result.trump_suit
        self.playing_team_id = tasmiya_result.playing_team_id
        self.defending_team_id = tasmiya_result.defending_team_id

        self.player_bids = ["Pass", "Pass", "Pass", "Pass"]
        for player_id, bid_value in tasmiya_result.bid_history:
            if bid_value is not None:
                self.player_bids[player_id] = str(bid_value)

        self.round.state.trump_suit = self.trump_suit
        self.round.state.winning_bidder_id = tasmiya_result.winning_bidder_id
        self.first_shooter_player_id = tasmiya_result.winning_bidder_id
        self.round.next_leading_player_id = tasmiya_result.winning_bidder_id

        self._last_tasmiya_result = tasmiya_result

        if self.step_mode:
            # Show bids step by step.
            self._bid_reveal_index = 0
            self._bid_order = list(tasmiya_result.bid_history)
            # Clear bids and other info but keep Qabool visible.
            for i in range(4):
                self.app.set_player_bid(i, "-")
            self.set_shota_info_safe(
                trump="-",
                qabool=f"Player {self.sahib_al_qabool_id + 1}",
                bid="-",
                first_shooter="-",
            )
            self.phase = "bidding_step"
            self._show_next_bid()
        else:
            # Auto: show all bids at once.
            self._update_gui_after_tasmiya()
            for i, bid_text in enumerate(self.player_bids):
                self.app.set_player_bid(i, bid_text)
            self.phase = "setup_ready"
            self._schedule(700, self.continue_simulation)

    def _init_step_bidding(self) -> None:
        """Legacy — no longer used directly. Kept for compatibility."""
        pass

    def _show_next_bid(self) -> None:
        """Reveal the next bid in the sequence."""
        if self._bid_reveal_index >= len(self._bid_order):
            # All bids revealed.
            is_dak = getattr(self, "_tasmiya_is_dak", False)

            if is_dak:
                # Show Dak message and go to dak_shown.
                deal_num = getattr(self, "_deal_number_counter", 1)
                self.app.set_current_trick(
                    f"دك — DAK #{self.game_state.pass_based_dak_count}\n"
                    f"Deal {deal_num}: All passed\n"
                    f"Sahib Al-Qabool (P{self.sahib_al_qabool_id + 1}) declares Dak\n"
                    f"Press Next to re-deal"
                )
                self.phase = "dak_shown"
                self._tasmiya_is_dak = False
            else:
                # Normal bid — show final info and move to setup_ready.
                self._update_gui_after_tasmiya()
                self.app.set_current_trick(
                    f"Bidding complete\n"
                    f"Winner: Player {self.first_shooter_player_id + 1}\n"
                    f"Bid: {self.bid_value} | Trump: {self.trump_suit.name}"
                )
                self.phase = "setup_ready"
            return

        player_id, bid_value = self._bid_order[self._bid_reveal_index]
        self._bid_reveal_index += 1

        # Show this player's bid.
        is_qabool = (player_id == self.sahib_al_qabool_id)

        if bid_value is not None:
            self.app.set_player_bid(player_id, str(bid_value))
            bid_text = f"Bid {bid_value}"
        else:
            # Sahib Al-Qabool passing = declaring Dak. Others just pass.
            if is_qabool and self._last_tasmiya_result and self._last_tasmiya_result.is_dak:
                self.app.set_player_bid(player_id, "Dak")
                bid_text = "Dak (دك)"
            else:
                self.app.set_player_bid(player_id, "Pass")
                bid_text = "Pass"

        # Highlight the bidding player.
        for i in range(4):
            if i == player_id:
                self.set_player_status_safe(i, f"Bidding: {bid_text}",
                                            is_qabool=(i == self.sahib_al_qabool_id))
            else:
                self.set_player_status_safe(i, "Waiting",
                                            is_qabool=(i == self.sahib_al_qabool_id))

        self.app.set_status(f"Player {player_id + 1}: {bid_text}. Press Next.")
        self.app.set_current_trick(
            f"Al-Tasmiya (Bidding)\n"
            f"Player {player_id + 1}: {bid_text}"
        )

    # ---------------------------------------------------------
    # Step-by-step card play
    # ---------------------------------------------------------

    def _play_next_card_step(self) -> None:
        """
        In step mode, play one card at a time within a trick.
        Uses a sub-state to track which card in the current trick we're on.
        """
        if not self.running:
            return

        # If we just showed the trick result, clear it and move on.
        if hasattr(self, "_trick_just_finished") and self._trick_just_finished:
            self._trick_just_finished = False
            # Check if game is over.
            if self.current_trick_number >= 13:
                self.finish_simulation()
                return
            # Otherwise fall through to start a new trick.

        # If we have a pending trick in progress, play the next card.
        if hasattr(self, "_trick_in_progress") and self._trick_in_progress:
            self._play_one_card_in_trick()
            return

        # Start a new trick.
        if self.current_trick_number >= 13:
            self.finish_simulation()
            return

        self.current_trick_number += 1
        self.app.clear_played_cards()
        self.show_player_hands()

        from environments.wist.trick import Trick

        if self.round.state is None or self.round.state.trump_suit is None:
            self.finish_simulation()
            return

        leader_id = self.round.next_leading_player_id
        self.round.state.current_trick = Trick(leading_player_id=leader_id)

        self._trick_play_order = [(leader_id + i) % 4 for i in range(4)]
        self._trick_card_index = 0
        self._trick_in_progress = True
        self._trick_just_finished = False

        # Play the first card immediately.
        self._play_one_card_in_trick()

    def _play_one_card_in_trick(self) -> None:
        """Play a single card and pause."""
        if self._trick_card_index >= 4:
            # Trick complete — resolve winner.
            self._finish_current_trick()
            return

        # Safety: ensure we have a current trick.
        if self.round.state.current_trick is None:
            self._trick_in_progress = False
            self._trick_just_finished = True
            return

        player_id = self._trick_play_order[self._trick_card_index]

        # Get the agent's action.
        observation = self.environment.observe(player_id)
        action = self.agents[player_id].act(observation)
        self.environment.apply_action(action)

        self._trick_card_index += 1

        # Reveal trump after first card of first trick.
        if self.current_trick_number == 1 and self._trick_card_index == 1:
            self.set_shota_info_safe(
                trump=self._format_trump(self.trump_suit),
                qabool=f"Player {self.qabool_player_index + 1}",
                bid=str(self.bid_value),
                first_shooter=f"Player {self.first_shooter_player_id + 1}",
            )

        # Generate explanation for why this card was played.
        card_text = self.format_card_short(action.card)
        reason = self._explain_card_play(player_id, action.card, observation)

        # Show the played card.
        self.app.set_played_cards(
            self._get_current_trick_cards()
        )

        # Update player hand (card removed).
        self.show_player_hands()

        # Highlight who just played.
        for i in range(4):
            if i == player_id:
                self.set_player_status_safe(i, f"Played {card_text}",
                                            is_qabool=(i == self.qabool_player_index),
                                            is_first_shooter=(i == self.first_shooter_player_id))
            else:
                self.set_player_status_safe(i, "Waiting",
                                            is_qabool=(i == self.qabool_player_index),
                                            is_first_shooter=(i == self.first_shooter_player_id))

        if self._trick_card_index < 4:
            next_player = self._trick_play_order[self._trick_card_index]
            self.app.set_status(
                f"P{player_id + 1} → {card_text} ({reason}). Next: P{next_player + 1}."
            )
            self.app.set_current_trick(
                f"Trick {self.current_trick_number}\n"
                f"P{player_id + 1}: {card_text}\n"
                f"{reason}"
            )
        else:
            # All 4 cards played — show them, next press will resolve.
            self.app.set_status(f"P{player_id + 1} → {card_text} ({reason}). Press Next for result.")
            self.app.set_current_trick(
                f"Trick {self.current_trick_number}\n"
                f"P{player_id + 1}: {card_text}\n"
                f"{reason}"
            )

    # _explain_card_play is in controller_helpers.py (ControllerHelpersMixin)

    def _get_current_trick_cards(self) -> list[tuple[int, str]]:
        """Get all cards played so far in the current trick."""
        trick = self.round.state.current_trick
        if trick is None:
            return []
        result = []
        for played_card in trick.played_cards:
            result.append((played_card.player_id, self.format_card_short(played_card.card)))
        return result

    def _finish_current_trick(self) -> None:
        """Resolve the trick winner and clean up."""
        from environments.wist.rules import trick_winner

        self._trick_in_progress = False
        self._trick_just_finished = True

        trick = self.round.state.current_trick

        winner = trick_winner(trick=trick, trump_suit=self.round.state.trump_suit)

        self.round.state.completed_tricks.append(trick)
        self.round.state.current_trick = None
        self.round.next_leading_player_id = winner

        winning_team_index = self.get_team_index(winner)
        self.team_tricks[winning_team_index] += 1
        self.player_tricks_won[winner] += 1

        self.set_player_status_safe(
            winner, "Won last trick",
            is_qabool=(winner == self.qabool_player_index),
            is_first_shooter=(winner == self.first_shooter_player_id),
        )

        try:
            self.app.set_tricks_won(self.player_tricks_won)
        except (AttributeError, TypeError):
            pass

        self.app.set_current_trick(
            f"Trick {self.current_trick_number} — Winner: Player {winner + 1}\n"
            f"Team 1: {self.team_tricks[0]} | Team 2: {self.team_tricks[1]}"
        )
        self.app.set_status(
            f"Player {winner + 1} won trick {self.current_trick_number}. Press Next."
        )

    def pause(self) -> None:
        if not self.running:
            self.app.log("No running simulation to pause.")
            return

        self.paused = True
        self.app.set_status("Simulation paused.")
        self.app.log("Simulation paused. Press Continue to resume.")

    def stop(self) -> None:
        if not self.running:
            self.app.log("Simulation is not running.")
            return

        self.running = False
        self.paused = False
        self.phase = "stopped"

        self.app.set_status("Simulation stopped.")
        self.app.set_current_trick("Simulation stopped.")
        self.app.log("Simulation stopped.")

    # ---------------------------------------------------------
    # Trick execution
    # ---------------------------------------------------------

    def play_next_trick_auto(self) -> None:
        """
        Auto mode:
        play one trick, then schedule the next trick.
        """

        if not self.running or self.paused:
            return

        self.play_one_trick()

        if self.running and self.phase == "playing_tricks":
            self._schedule(700, self.play_next_trick_auto)
        elif self.running and self.phase == "shota_ended":
            if self._auto_game_mode:
                # Auto Game: continue to next shota immediately.
                self._schedule(700, self._auto_next_shota)
            else:
                # Auto Shota: pause between shotas — wait for user.
                pass

    def play_next_trick_step(self) -> None:
        """
        Step mode:
        play one trick, then wait for Continue.
        """

        if not self.running:
            return

        self.play_one_trick()

        if self.running and self.phase == "playing_tricks":
            self.app.set_status("Paused after trick. Press Continue.")

    def play_one_trick(self) -> None:
        """
        Play exactly one trick.
        No automatic scheduling happens here.
        """

        if not self.running:
            return

        if self.phase != "playing_tricks":
            return

        if self.current_trick_number >= 13:
            self.finish_simulation()
            return

        self.current_trick_number += 1

        for player_index in range(4):
            self.set_player_status_safe(
                player_index,
                "Waiting",
                is_qabool=(player_index == self.qabool_player_index),
                is_first_shooter=(player_index == self.first_shooter_player_id),
            )
            self.app.set_player_bid(player_index, self.player_bids[player_index])

        self.app.clear_played_cards()
        self.show_player_hands()

        self.app.set_current_trick(f"Playing trick {self.current_trick_number}...")
        self.app.log("")
        self.app.log(f"--- Trick {self.current_trick_number} ---")

        trick_details = self.engine.play_trick_details(
            round_=self.round,
            environment=self.environment,
            agents=self.agents,
        )

        winner = trick_details["winner"]
        leader = trick_details["leader"]
        play_order = trick_details["play_order"]
        trick = trick_details["trick"]

        # Reveal trump after the first trick (first card reveals it).
        if self.current_trick_number == 1:
            self.set_shota_info_safe(
                trump=self._format_trump(self.trump_suit),
                qabool=f"Player {self.qabool_player_index + 1}",
                bid=str(self.bid_value),
                first_shooter=f"Player {self.first_shooter_player_id + 1}",
            )

        self.app.log(f"Leader: Player {leader + 1}")
        self.app.log(
            "Play order: "
            + " → ".join(f"Player {player_id + 1}" for player_id in play_order)
        )

        winning_team_index = self.get_team_index(winner)
        self.team_tricks[winning_team_index] += 1
        self.player_tricks_won[winner] += 1

        self.show_player_hands()
        self.log_trick_cards(trick)

        self.app.log(f"Player {winner + 1} won the trick.")
        self.app.log(f"Team 1 tricks: {self.team_tricks[0]}")
        self.app.log(f"Team 2 tricks: {self.team_tricks[1]}")

        self.set_player_status_safe(
            winner,
            "Won last trick",
            is_qabool=(winner == self.qabool_player_index),
            is_first_shooter=(winner == self.first_shooter_player_id),
        )

        # Update won-tricks display.
        try:
            self.app.set_tricks_won(self.player_tricks_won)
        except (AttributeError, TypeError):
            pass

        self.app.set_current_trick(
            f"Trick {self.current_trick_number} complete\n"
            f"Winner: Player {winner + 1}\n"
            f"Team 1: {self.team_tricks[0]} | Team 2: {self.team_tricks[1]}"
        )

        if self.current_trick_number >= 13:
            self.finish_simulation()

    # ---------------------------------------------------------
    # Dealing and Dak
    # ---------------------------------------------------------

    def _deal_and_check_dak(self) -> None:
        """
        Deal cards and check for card-based Dak.
        Re-deals if Dak is detected (up to 10 attempts).
        Card-based Dak in the first Shota does NOT count toward the 5 Shotas.
        """

        if not hasattr(self, "_deal_number"):
            self._deal_number = 1
        else:
            self._deal_number += 1

        self.round = Round(self.players)
        self.round.deal()

        try:
            self.app.set_deal_number(self._deal_number)
        except (AttributeError, TypeError):
            pass

        redeal_count = 0
        max_redeals = 10

        while self.round.has_card_based_dak() and redeal_count < max_redeals:
            dak_player = self.round.first_card_based_dak_player_id()

            self.app.set_current_trick(
                f"دك — Card-Based DAK!\n"
                f"Player {dak_player + 1} declares Dak\n"
                f"Re-dealing..."
            )

            # After the first Shota, card-based Dak counts as a Shota.
            if self.game_state.completed_shotas > 0:
                self.game_state.apply_dak(DakType.CARD_BASED)

            self._deal_number += 1
            self.round = Round(self.players)
            self.round.deal()
            redeal_count += 1

            try:
                self.app.set_deal_number(self._deal_number)
            except (AttributeError, TypeError):
                pass

        self.environment = WistEnvironment(self.round.state)

    # ---------------------------------------------------------
    # Shota setup
    # ---------------------------------------------------------

    def prepare_shota_setup(self) -> None:
        """
        Run the Al-Tasmiya (bidding) phase using the TasmiyaEngine.

        The engine asks each agent for bid/pass decisions in the correct
        counter-clockwise order, then Sahib Al-Qabool decides.

        Handles pass-based Dak with the 2-per-game limit.
        """

        tasmiya_result = self.shota_setup_engine.run(
            players=self.players,
            agents=self.agents,
            sahib_al_qabool_id=self.sahib_al_qabool_id,
        )

        if tasmiya_result.is_dak:
            # Pass-based Dak.
            if self.game_state.pass_based_dak_count < 2:
                # Allowed: apply Dak, rotate Qabool, re-deal.
                self.game_state.apply_dak(DakType.PASS_BASED)

                self.app.set_current_trick(
                    f"دك — DAK #{self.game_state.pass_based_dak_count}\n"
                    f"All players passed!\n"
                    f"Qabool rotates to Player {(self.sahib_al_qabool_id + 1) % 4 + 1}"
                )

                self.sahib_al_qabool_id = (self.sahib_al_qabool_id + 1) % 4

                self._deal_and_check_dak()
                self.show_player_hands()

                # Retry Tasmiya with new Qabool.
                tasmiya_result = self.shota_setup_engine.run(
                    players=self.players,
                    agents=self.agents,
                    sahib_al_qabool_id=self.sahib_al_qabool_id,
                )

                # If Dak again and still under limit, handle it.
                if tasmiya_result.is_dak and self.game_state.pass_based_dak_count < 2:
                    self.game_state.apply_dak(DakType.PASS_BASED)

                    self.app.set_current_trick(
                        f"دك — DAK #{self.game_state.pass_based_dak_count}\n"
                        f"All players passed again!\n"
                        f"Qabool rotates to Player {(self.sahib_al_qabool_id + 1) % 4 + 1}"
                    )

                    self.sahib_al_qabool_id = (self.sahib_al_qabool_id + 1) % 4

                    self._deal_and_check_dak()
                    self.show_player_hands()

                    tasmiya_result = self.shota_setup_engine.run(
                        players=self.players,
                        agents=self.agents,
                        sahib_al_qabool_id=self.sahib_al_qabool_id,
                    )

            if tasmiya_result.is_dak:
                # Third attempt or limit reached: force Qabool to play.
                # Bid is based on hand strength, not hardcoded to 7.
                from environments.wist.tasmiya_engine import max_bid_for_hand
                from collections import Counter

                qabool_hand = self.players[self.sahib_al_qabool_id].hand
                trump = determine_trump_suit(qabool_hand)

                suit_counts = Counter(c.suit for c in qabool_hand)
                longest_count = max(suit_counts.values())
                forced_bid = min(longest_count + 3, 13)
                forced_bid = max(7, forced_bid)

                self.qabool_player_index = self.sahib_al_qabool_id
                self.bid_value = forced_bid
                self.trump_suit = trump
                self.playing_team_id = self.players[self.sahib_al_qabool_id].team_id
                self.defending_team_id = 1 if self.playing_team_id == 0 else 0
                self.player_bids = ["Pass", "Pass", "Pass", "Pass"]
                self.player_bids[self.sahib_al_qabool_id] = f"{forced_bid} (forced)"

                self.round.state.trump_suit = self.trump_suit
                self.round.state.winning_bidder_id = self.sahib_al_qabool_id
                self.first_shooter_player_id = self.sahib_al_qabool_id
                self.round.next_leading_player_id = self.sahib_al_qabool_id

                self.app.set_current_trick(
                    f"DAK limit reached!\n"
                    f"Player {self.sahib_al_qabool_id + 1} forced to play\n"
                    f"Bid: {forced_bid} | Trump: {trump.name}"
                )

                self._update_gui_after_tasmiya()
                self._last_tasmiya_result = tasmiya_result
                return

        # Normal bidding result.
        self.qabool_player_index = tasmiya_result.sahib_al_qabool_id
        self.bid_value = tasmiya_result.winning_bid_value
        self.trump_suit = tasmiya_result.trump_suit
        self.playing_team_id = tasmiya_result.playing_team_id
        self.defending_team_id = tasmiya_result.defending_team_id

        # Build player bid display from history.
        self.player_bids = ["Pass", "Pass", "Pass", "Pass"]
        for player_id, bid_value in tasmiya_result.bid_history:
            if bid_value is not None:
                self.player_bids[player_id] = str(bid_value)

        # Set the round state for play.
        self.round.state.trump_suit = self.trump_suit
        self.round.state.winning_bidder_id = tasmiya_result.winning_bidder_id

        # The winning bidder leads the first trick.
        self.first_shooter_player_id = tasmiya_result.winning_bidder_id
        self.round.next_leading_player_id = tasmiya_result.winning_bidder_id

        self._log_tasmiya_result(tasmiya_result)
        self._update_gui_after_tasmiya()
        self._last_tasmiya_result = tasmiya_result

    def _log_tasmiya_result(self, tasmiya_result, forced: bool = False) -> None:
        """Log the Tasmiya result to the GUI."""

        self.app.log("")
        self.app.log("=== Al-Tasmiya (Bidding) ===")

        for player_id, bid_value in tasmiya_result.bid_history:
            if bid_value is not None:
                suffix = ""
                if player_id == tasmiya_result.sahib_al_qabool_id:
                    suffix = " (Sahib Al-Qabool)"
                self.app.log(f"Player {player_id + 1}: Bid {bid_value}{suffix}")
            else:
                self.app.log(f"Player {player_id + 1}: Pass")

        if forced:
            self.app.log(f"Forced play: Player {self.qabool_player_index + 1} must play.")
        else:
            self.app.log(f"Winning bidder: Player {tasmiya_result.winning_bidder_id + 1}")
            self.app.log(f"Playing team: Team {tasmiya_result.playing_team_id + 1}")
            self.app.log(f"Defending team: Team {tasmiya_result.defending_team_id + 1}")

        self.app.log(f"Bid: {self.bid_value}")
        self.app.log(f"Trump suit: {self.trump_suit.name if self.trump_suit else '-'}")

    def _update_gui_after_tasmiya(self) -> None:
        """Update the GUI with Tasmiya results.
        Trump is hidden until the first card is played (per rules)."""

        self.set_shota_info_safe(
            trump="? (hidden)",
            qabool=f"Player {self.qabool_player_index + 1}",
            bid=str(self.bid_value) if self.bid_value else "-",
            first_shooter=f"Player {self.first_shooter_player_id + 1}",
        )

        for player_index, bid_text in enumerate(self.player_bids):
            self.app.set_player_bid(player_index, bid_text)
            self.set_player_status_safe(
                player_index,
                "Qabool / Trump" if player_index == self.qabool_player_index else "Waiting",
                is_qabool=(player_index == self.qabool_player_index),
                is_first_shooter=(player_index == self.first_shooter_player_id),
            )

    # ---------------------------------------------------------
    # Finish
    # ---------------------------------------------------------

    def finish_simulation(self) -> None:
        # Prevent double-calling.
        if self.phase not in ("playing_tricks",):
            return

        # Guard: ensure we actually have 13 tricks worth of data.
        total_tricks = self.team_tricks[0] + self.team_tricks[1]
        if total_tricks != 13:
            return

        # --- Apply scoring ---
        team_tricks_dict = {0: self.team_tricks[0], 1: self.team_tricks[1]}

        # Check for Seek.
        seek_team = detect_seek(team_tricks_dict)
        if seek_team is not None:
            self.game_state.apply_seek(seek_team)
        elif self.playing_team_id is not None and self.defending_team_id is not None:
            score_delta = score_shota(
                playing_team_id=self.playing_team_id,
                defending_team_id=self.defending_team_id,
                bid=self.bid_value,
                playing_team_tricks=team_tricks_dict[self.playing_team_id],
                defending_team_tricks=team_tricks_dict[self.defending_team_id],
            )
            self.game_state.apply_shota_score(score_delta)

        # Update game score display.
        try:
            self.app.set_game_score(
                team_1=self.game_state.team_scores[0],
                team_2=self.game_state.team_scores[1],
                shotas=self.game_state.completed_shotas,
            )
        except (AttributeError, TypeError):
            pass

        # Build result message.
        bid_result = ""
        shota_winner_team = None
        if self.playing_team_id is not None and seek_team is None:
            if team_tricks_dict.get(self.playing_team_id, 0) >= self.bid_value:
                bid_result = f"Bid SUCCESS (+{team_tricks_dict[self.playing_team_id]})"
                shota_winner_team = self.playing_team_id
            else:
                bid_result = f"Bid FAILED (-{self.bid_value})"
                shota_winner_team = self.defending_team_id
        elif seek_team is not None:
            shota_winner_team = seek_team

        # Show which team won this Shota.
        try:
            if shota_winner_team is not None:
                self.app.shota_info_labels["winner"].config(text=f"Team {shota_winner_team + 1}")
        except (AttributeError, KeyError):
            pass

        # Determine if game is over or more Shotas to play.
        if self.game_state.is_finished():
            # Game over.
            self.running = False
            self.phase = "completed"
            winner = self.game_state.winner_team_id

            # If no team reached 25, the team with the higher score wins.
            if winner is None:
                if self.game_state.team_scores[0] > self.game_state.team_scores[1]:
                    winner = 0
                elif self.game_state.team_scores[1] > self.game_state.team_scores[0]:
                    winner = 1

            try:
                if winner is not None:
                    self.app.show_game_over(
                        winner_team=winner + 1,
                        score_1=self.game_state.team_scores[0],
                        score_2=self.game_state.team_scores[1],
                    )
                else:
                    # True draw.
                    self.app.show_game_over(
                        winner_team=0,
                        score_1=self.game_state.team_scores[0],
                        score_2=self.game_state.team_scores[1],
                    )
            except (AttributeError, TypeError):
                pass
        else:
            # More Shotas to play.
            self.phase = "shota_ended"
            self.sahib_al_qabool_id = (self.sahib_al_qabool_id + 1) % 4

            self.app.set_current_trick(
                f"Shota {self.game_state.completed_shotas} complete  {bid_result}\n"
                f"T1: {self.team_tricks[0]} tricks | T2: {self.team_tricks[1]} tricks\n"
                f"Score: {self.game_state.team_scores[0]} - {self.game_state.team_scores[1]}"
            )

            if self.step_mode:
                self.app.set_status(
                    f"Shota {self.game_state.completed_shotas}/5 done. Press Next for next Shota."
                )
            else:
                self.app.set_status(
                    f"Shota {self.game_state.completed_shotas}/5 done. Starting next..."
                )

    def _start_next_shota_auto(self) -> None:
        """Start the next Shota in auto mode."""
        if not self.running or self.game_state.is_finished():
            return
        self._reset_for_new_shota()
        self.continue_simulation()

    def _auto_next_shota(self) -> None:
        """Auto game mode: start next shota without user input."""
        if not self.running or self.game_state.is_finished():
            return
        self._reset_for_new_shota()
        self.continue_simulation()

    def _start_next_shota_step(self) -> None:
        """Start the next Shota in step mode."""
        if not self.running or self.game_state.is_finished():
            return
        self._reset_for_new_shota()
        # Show deal, then wait for Next.
        self.app.set_status("New Shota dealt. Press Next.")

    def _reset_for_new_shota(self) -> None:
        """Reset per-Shota state for the next Shota."""
        self.current_trick_number = 0
        self.team_tricks = [0, 0]
        self.player_tricks_won = [0, 0, 0, 0]
        self.trump_suit = None
        self.qabool_player_index = None
        self.bid_value = None
        self.player_bids = ["-", "-", "-", "-"]
        self.playing_team_id = None
        self.defending_team_id = None
        self._trick_in_progress = False
        self._trick_just_finished = False
        self._deal_number = 0  # Will be incremented to 1 in _deal_and_check_dak.
        self._deal_number_counter = 1

        self.app.reset_player_statuses()
        self.app.clear_played_cards()

        # Clear winner label for the new Shota.
        try:
            self.app.shota_info_labels["winner"].config(text="—")
        except (AttributeError, KeyError):
            pass

        self._deal_and_check_dak()
        self.show_player_hands()
        self.phase = "dealt"

        # Show current Shota number immediately.
        try:
            current_shota = self.game_state.completed_shotas + 1
            self.app.set_game_score(
                team_1=self.game_state.team_scores[0],
                team_2=self.game_state.team_scores[1],
                shotas=current_shota,
            )
        except (AttributeError, TypeError):
            pass

    # ---------------------------------------------------------
    # Helper methods are in controller_helpers.py (ControllerHelpersMixin)
    # ---------------------------------------------------------
    # Helper methods are in controller_helpers.py (ControllerHelpersMixin)
    # ---------------------------------------------------------
