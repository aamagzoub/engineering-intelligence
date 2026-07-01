from environments.wist.actions import PlayCardAction
from environments.wist.observation import WistObservation
from environments.wist.round_state import RoundState
from intelligence.core.action import Action
from intelligence.core.environment import Environment


class WistEnvironment(Environment):
    """
    Wist environment.

    Owns the true round state and exposes observations to players.
    """

    def __init__(self, state: RoundState) -> None:
        self.state = state

    def observe(self, player_id: int) -> WistObservation:
        player = self.state.get_player(player_id)

        return WistObservation(
            player_id=player.player_id,
            hand=list(player.hand),
            current_trick=self.state.current_trick,
            trump_suit=self.state.trump_suit,
            played_cards=list(self.state.played_cards),
        )

    def apply_action(self, action: Action) -> None:
        if not isinstance(action, PlayCardAction):
            raise TypeError("WistEnvironment only supports PlayCardAction for now.")

        player = self.state.get_player(action.player_id)
        card = player.play_card(action.card)

        if self.state.current_trick is None:
            raise ValueError("No current trick exists.")

        self.state.current_trick.play_card(
            player_id=player.player_id,
            card=card,
        )

        self.state.played_cards.append(card)