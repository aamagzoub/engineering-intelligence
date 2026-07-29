from environments.wist.bidding import (
    Bid,
    BidResult,
    Pass,
    validate_opening_bid,
    validate_regular_bid,
    validate_sahib_al_qabool_bid,
)


class BiddingEngine:
    """
    Controls the bidding phase of one Shota.
    """

    def __init__(self) -> None:
        self.highest_bid: Bid | None = None
        self.passed_player_ids: list[int] = []

    def apply_bid(
        self,
        bid: Bid,
        is_sahib_al_qabool: bool = False,
    ) -> None:

        if self.highest_bid is None:
            validate_opening_bid(bid)
        elif is_sahib_al_qabool:
            validate_sahib_al_qabool_bid(
                bid,
                self.highest_bid,
            )
        else:
            validate_regular_bid(
                bid,
                self.highest_bid,
            )

        self.highest_bid = bid

    def apply_pass(
        self,
        action: Pass,
    ) -> None:
        self.passed_player_ids.append(action.player_id)

    def result(self) -> BidResult:
        return BidResult(
            highest_bid=self.highest_bid,
            passed_player_ids=self.passed_player_ids,
        )