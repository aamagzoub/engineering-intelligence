from dataclasses import dataclass


@dataclass(frozen=True)
class Bid:
    player_id: int
    value: int

    def __post_init__(self) -> None:
        if self.value < 7 or self.value > 13:
            raise ValueError("Bid must be between 7 and 13.")


@dataclass(frozen=True)
class Pass:
    player_id: int


@dataclass(frozen=True)
class BidResult:
    highest_bid: Bid | None
    passed_player_ids: list[int]

    def all_passed(self) -> bool:
        return self.highest_bid is None


def validate_regular_bid(
    bid: Bid,
    current_highest_bid: Bid | None,
) -> None:
    if current_highest_bid is None:
        return

    if bid.value <= current_highest_bid.value:
        raise ValueError("Regular bid must be higher than the current highest bid.")

def validate_sahib_al_qabool_bid(
    bid: Bid,
    current_highest_bid: Bid | None,
) -> None:
    """
    Sahib Al-Qabool may match or exceed the current highest bid.
    """

    if current_highest_bid is None:
        return

    if bid.value < current_highest_bid.value:
        raise ValueError(
            "Sahib Al-Qabool bid must match or exceed the current highest bid."
        )

def validate_opening_bid(bid: Bid) -> None:
    """
    The first actual bid cannot exceed 11.
    Passes do not count as opening bids.
    """

    if bid.value > 11:
        raise ValueError("Opening bid cannot exceed 11.")