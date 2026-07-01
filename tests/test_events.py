from environments.wist.events import card_based_dak_event, trick_completed_event
from intelligence.core.event import Event


def test_event_gets_id_and_timestamp():
    event = Event(
        event_type="test.event",
        payload={"value": 1},
    )

    assert event.event_id != ""
    assert event.created_at is not None
    assert event.event_type == "test.event"
    assert event.payload == {"value": 1}


def test_card_based_dak_event():
    event = card_based_dak_event(
        player_id=2,
        reason="8 or more cards in one suit",
    )

    assert event.event_type == "wist.card_based_dak"
    assert event.payload["player_id"] == 2
    assert event.payload["reason"] == "8 or more cards in one suit"


def test_trick_completed_event():
    event = trick_completed_event(
        winner_player_id=3,
        winner_team_id=1,
    )

    assert event.event_type == "wist.trick_completed"
    assert event.payload["winner_player_id"] == 3
    assert event.payload["winner_team_id"] == 1