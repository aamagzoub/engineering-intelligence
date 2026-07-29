from intelligence.core.event import Event


def card_based_dak_event(player_id: int, reason: str) -> Event:
    return Event(
        event_type="wist.card_based_dak",
        payload={
            "player_id": player_id,
            "reason": reason,
        },
    )


def trick_completed_event(winner_player_id: int, winner_team_id: int) -> Event:
    return Event(
        event_type="wist.trick_completed",
        payload={
            "winner_player_id": winner_player_id,
            "winner_team_id": winner_team_id,
        },
    )