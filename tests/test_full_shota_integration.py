"""
Integration test: run a full Shota using the TasmiyaEngine + PlayingEngine.

This replicates the controller's flow without the GUI.
"""

from agents.random.random_agent import RandomAgent
from environments.wist.environment import WistEnvironment
from environments.wist.playing_engine import PlayingEngine
from environments.wist.round import Round
from environments.wist.scoring import score_shota, detect_seek
from environments.wist.setup import create_standard_players
from environments.wist.tasmiya_engine import TasmiyaEngine


def test_full_shota_with_tasmiya_engine():
    """
    Run 20 full Shotas end-to-end to verify the integrated flow.
    Each Shota: deal → Tasmiya → 13 tricks → score.
    """

    for iteration in range(20):
        players = create_standard_players()
        agents = [RandomAgent(), RandomAgent(), RandomAgent(), RandomAgent()]

        round_ = Round(players)
        round_.deal()

        # Handle card-based Dak by re-dealing (up to 5 attempts).
        redeal_count = 0
        while round_.has_card_based_dak() and redeal_count < 5:
            round_ = Round(players)
            round_.deal()
            redeal_count += 1

        if round_.has_card_based_dak():
            # Extremely unlikely after 5 re-deals, skip this iteration.
            continue

        # Run Al-Tasmiya.
        tasmiya_engine = TasmiyaEngine()
        sahib_al_qabool_id = 0

        tasmiya_result = tasmiya_engine.run(
            players=players,
            agents=agents,
            sahib_al_qabool_id=sahib_al_qabool_id,
        )

        if tasmiya_result.is_dak:
            # All passed and Qabool declared Dak — valid outcome, skip play.
            continue

        # Set up the round for play.
        round_.state.trump_suit = tasmiya_result.trump_suit
        round_.state.winning_bidder_id = tasmiya_result.winning_bidder_id
        round_.next_leading_player_id = tasmiya_result.winning_bidder_id

        environment = WistEnvironment(round_.state)

        # Play 13 tricks.
        engine = PlayingEngine()
        team_tricks = engine.play_shota(round_, environment, agents)

        # Validate.
        assert team_tricks[0] + team_tricks[1] == 13
        assert len(round_.state.completed_tricks) == 13
        assert len(round_.state.played_cards) == 52

        for player in players:
            assert len(player.hand) == 0

        # First card of first trick must be trump.
        first_trick = round_.state.completed_tricks[0]
        first_card_played = first_trick.played_cards[0]
        assert first_card_played.player_id == tasmiya_result.winning_bidder_id
        assert first_card_played.card.suit == tasmiya_result.trump_suit

        # Score the Shota.
        score = score_shota(
            playing_team_id=tasmiya_result.playing_team_id,
            defending_team_id=tasmiya_result.defending_team_id,
            bid=tasmiya_result.winning_bid_value,
            playing_team_tricks=team_tricks[tasmiya_result.playing_team_id],
            defending_team_tricks=team_tricks[tasmiya_result.defending_team_id],
        )

        # Score should be a valid dict with both team IDs.
        assert tasmiya_result.playing_team_id in score
        assert tasmiya_result.defending_team_id in score

        # Check Seek detection.
        seek_team = detect_seek(team_tricks)
        if seek_team is not None:
            assert team_tricks[seek_team] == 13


def test_trump_is_bidders_longest_suit():
    """
    Verify that the trump suit determined by TasmiyaEngine
    is always the winning bidder's longest suit.
    """
    from collections import Counter

    for _ in range(30):
        players = create_standard_players()
        agents = [RandomAgent(), RandomAgent(), RandomAgent(), RandomAgent()]

        round_ = Round(players)
        round_.deal()

        if round_.has_card_based_dak():
            continue

        tasmiya_engine = TasmiyaEngine()
        result = tasmiya_engine.run(
            players=players,
            agents=agents,
            sahib_al_qabool_id=0,
        )

        if result.is_dak:
            continue

        # Check that trump is the winning bidder's longest suit.
        bidder_hand = players[result.winning_bidder_id].hand
        suit_counts = Counter(card.suit for card in bidder_hand)
        longest_count = max(suit_counts.values())
        longest_suits = [s for s, c in suit_counts.items() if c == longest_count]

        assert result.trump_suit in longest_suits, (
            f"Trump {result.trump_suit} not in longest suits {longest_suits} "
            f"for player {result.winning_bidder_id}"
        )
