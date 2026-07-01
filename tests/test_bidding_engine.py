from environments.wist.bidding import Bid, Pass
from environments.wist.bidding_engine import BiddingEngine


def test_bidding_engine_accepts_first_bid():
    engine = BiddingEngine()

    engine.apply_bid(
        Bid(0, 8)
    )

    assert engine.highest_bid.value == 8


def test_bidding_engine_records_pass():
    engine = BiddingEngine()

    engine.apply_pass(
        Pass(2)
    )

    assert engine.passed_player_ids == [2]


def test_bidding_engine_returns_bid_result():
    engine = BiddingEngine()

    engine.apply_bid(
        Bid(0, 8)
    )

    result = engine.result()

    assert result.highest_bid.value == 8