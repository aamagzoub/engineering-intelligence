import pytest

from environments.wist.bidding import Bid, BidResult, Pass
from environments.wist.bidding import validate_regular_bid
from environments.wist.bidding import validate_sahib_al_qabool_bid
from environments.wist.bidding import validate_opening_bid

def test_bid_stores_player_and_value():
    bid = Bid(player_id=1, value=8)

    assert bid.player_id == 1
    assert bid.value == 8


def test_bid_must_be_between_7_and_13():
    with pytest.raises(ValueError):
        Bid(player_id=1, value=6)

    with pytest.raises(ValueError):
        Bid(player_id=1, value=14)


def test_pass_stores_player():
    action = Pass(player_id=2)

    assert action.player_id == 2


def test_bid_result_knows_when_all_passed():
    result = BidResult(
        highest_bid=None,
        passed_player_ids=[0, 1, 2],
    )

    assert result.all_passed() is True


def test_bid_result_knows_when_there_is_bid():
    bid = Bid(player_id=0, value=9)

    result = BidResult(
        highest_bid=bid,
        passed_player_ids=[1, 2],
    )

    assert result.all_passed() is False
    assert result.highest_bid == bid

def test_regular_bid_can_start_bidding():
    validate_regular_bid(
        bid=Bid(player_id=0, value=7),
        current_highest_bid=None,
    )


def test_regular_bid_must_be_higher_than_current_highest():
    current = Bid(player_id=0, value=8)

    validate_regular_bid(
        bid=Bid(player_id=1, value=9),
        current_highest_bid=current,
    )

    with pytest.raises(ValueError):
        validate_regular_bid(
            bid=Bid(player_id=2, value=8),
            current_highest_bid=current,
        )

def test_sahib_al_qabool_can_match_highest_bid():
    current = Bid(player_id=1, value=9)

    validate_sahib_al_qabool_bid(
        bid=Bid(player_id=0, value=9),
        current_highest_bid=current,
    )


def test_sahib_al_qabool_can_increase_highest_bid():
    current = Bid(player_id=1, value=9)

    validate_sahib_al_qabool_bid(
        bid=Bid(player_id=0, value=10),
        current_highest_bid=current,
    )


def test_sahib_al_qabool_cannot_bid_below_highest_bid():
    current = Bid(player_id=1, value=9)

    with pytest.raises(ValueError):
        validate_sahib_al_qabool_bid(
            bid=Bid(player_id=0, value=8),
            current_highest_bid=current,
        )

def test_opening_bid_can_be_11_or_lower():
    validate_opening_bid(Bid(player_id=0, value=11))


def test_opening_bid_cannot_exceed_11():
    with pytest.raises(ValueError):
        validate_opening_bid(Bid(player_id=0, value=12))