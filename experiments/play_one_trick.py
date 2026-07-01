from agents.random.random_agent import RandomAgent
from environments.wist.actions import PlayCardAction
from environments.wist.environment import WistEnvironment
from environments.wist.player import Player
from environments.wist.round_state import RoundState
from environments.wist.rules import trick_winner
from environments.wist.trick import Trick
from intelligence.core.cards.card import Card
from intelligence.core.cards.rank import Rank
from intelligence.core.cards.suit import Suit


def main() -> None:
    players = {
        0: Player(player_id=0, team_id=0, hand=[
            Card(Suit.SPADES, Rank.ACE),
            Card(Suit.HEARTS, Rank.TWO),
        ]),
        1: Player(player_id=1, team_id=1, hand=[
            Card(Suit.SPADES, Rank.KING),
            Card(Suit.CLUBS, Rank.TWO),
        ]),
        2: Player(player_id=2, team_id=0, hand=[
            Card(Suit.SPADES, Rank.QUEEN),
            Card(Suit.DIAMONDS, Rank.TWO),
        ]),
        3: Player(player_id=3, team_id=1, hand=[
            Card(Suit.SPADES, Rank.JACK),
            Card(Suit.HEARTS, Rank.THREE),
        ]),
    }

    state = RoundState(
        players=players,
        trump_suit=Suit.HEARTS,
        current_trick=Trick(leading_player_id=0),
    )

    environment = WistEnvironment(state)
    agent = RandomAgent()

    for player_id in [0, 1, 2, 3]:
        observation = environment.observe(player_id)
        action = agent.act(observation)

        if not isinstance(action, PlayCardAction):
            raise TypeError("Expected PlayCardAction.")

        environment.apply_action(action)

        print(f"Player {player_id} played {action.card}")

    winner = trick_winner(
        trick=state.current_trick,
        trump_suit=state.trump_suit,
    )

    print(f"Trick winner: Player {winner}")


if __name__ == "__main__":
    main()