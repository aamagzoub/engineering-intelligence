import random
from dataclasses import dataclass

from environments.wist.bidding import Bid, Pass
from environments.wist.bidding_engine import BiddingEngine
from environments.wist.player import Player
from intelligence.core.cards.suit import Suit


@dataclass(frozen=True)
class PlayerBidDisplay:
    player_id: int
    text: str
    is_qabool: bool = False


@dataclass(frozen=True)
class ShotaSetup:
    """
    Setup information before one Shota starts.
    """

    qabool_player_id: int
    playing_team_id: int
    defending_team_id: int
    bid: int
    trump_suit: Suit
    player_bids: list[PlayerBidDisplay]


class RandomShotaSetupEngine:
    """
    Temporary Shota setup engine.

    Uses the real Bid, Pass, and BiddingEngine classes,
    but still chooses decisions randomly for now.
    """

    def create_setup(self, players: list[Player]) -> ShotaSetup:
        if len(players) != 4:
            raise ValueError("A Wist Shota requires exactly 4 players.")

        bidding_engine = BiddingEngine()

        qabool_player = random.choice(players)

        # Opening bid cannot exceed 11 according to your current rule.
        bid_value = random.choice([7, 8, 9, 10, 11])

        trump_suit = random.choice(
            [
                Suit.CLUBS,
                Suit.DIAMONDS,
                Suit.HEARTS,
                Suit.SPADES,
            ]
        )

        player_bids: list[PlayerBidDisplay] = []

        for player in players:
            if player.player_id == qabool_player.player_id:
                bid = Bid(
                    player_id=player.player_id,
                    value=bid_value,
                )

                bidding_engine.apply_bid(
                    bid,
                    is_sahib_al_qabool=True,
                )

                player_bids.append(
                    PlayerBidDisplay(
                        player_id=player.player_id,
                        text=str(bid_value),
                        is_qabool=True,
                    )
                )
            else:
                pass_action = Pass(player_id=player.player_id)
                bidding_engine.apply_pass(pass_action)

                player_bids.append(
                    PlayerBidDisplay(
                        player_id=player.player_id,
                        text="Pass",
                        is_qabool=False,
                    )
                )

        result = bidding_engine.result()

        if result.highest_bid is None:
            raise ValueError("Shota setup failed: all players passed.")

        playing_team_id = qabool_player.team_id
        defending_team_id = 1 if playing_team_id == 0 else 0

        return ShotaSetup(
            qabool_player_id=qabool_player.player_id,
            playing_team_id=playing_team_id,
            defending_team_id=defending_team_id,
            bid=result.highest_bid.value,
            trump_suit=trump_suit,
            player_bids=player_bids,
        )