"""
Sudanese Hearts — Training & Play

Train the discovery agent to learn Hearts from scratch (no rules given).

Usage:
    python run_hearts.py train --episodes 5000
    python run_hearts.py train --episodes 10000 --model saved_model.json
    python run_hearts.py watch                          (watch one game trick by trick)
    python run_hearts.py watch --model hearts_model.json (watch trained agent play)
    python run_hearts.py play --model agents/hearts_discovery/hearts_model.json
    python run_hearts.py stats --model agents/hearts_discovery/hearts_model.json
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.hearts_discovery.discovery_agent import DiscoveryAgent
from agents.hearts_discovery.random_hearts_agent import RandomHeartsAgent
from environments.hearts.game import HeartsGame


def train(episodes: int, model_path: str | None = None, save_path: str | None = None):
    """
    Train the discovery agent by playing against random agents.

    The discovery agent is player 0. Players 1-3 are random baselines.
    """
    if save_path is None:
        save_path = "agents/hearts_discovery/hearts_model.json"

    # Create agents.
    discovery = DiscoveryAgent(epsilon=0.5, alpha=0.1, gamma=0.95, training=True)

    # Load existing model if provided.
    if model_path:
        try:
            discovery.load(model_path)
            print(f"Loaded model from {model_path}")
            print(f"  Episodes previously trained: {discovery.episodes_trained}")
        except FileNotFoundError:
            print(f"Model not found at {model_path}, starting fresh.")

    random_agents = [RandomHeartsAgent() for _ in range(3)]
    agents = [discovery, random_agents[0], random_agents[1], random_agents[2]]

    # Training metrics.
    scores_history: list[float] = []
    wins = 0
    total_games = 0

    print(f"\n{'='*60}")
    print(f"  Sudanese Hearts — Discovery Agent Training")
    print(f"  Episodes: {episodes}")
    print(f"  Agent knows: NOTHING (only legal moves + reward)")
    print(f"{'='*60}\n")

    batch_size = 100
    for episode in range(1, episodes + 1):
        # Play one full game (5 shotas).
        game = HeartsGame(agents=agents)
        result = game.play()

        # Get discovery agent's score for each shota.
        for shota_result in result.shota_results:
            agent_score = shota_result.scores.get(0, 0)
            discovery.reward(agent_score)
            scores_history.append(agent_score)

        # Track wins.
        total_games += 1
        if result.winner_id == 0:
            wins += 1

        # Decay epsilon over time (explore less as we learn).
        if episode % 200 == 0 and discovery.epsilon > 0.1:
            discovery.epsilon *= 0.95

        # Print progress every batch.
        if episode % batch_size == 0:
            recent_scores = scores_history[-batch_size * 5:]  # Last batch × 5 shotas
            avg_score = sum(recent_scores) / len(recent_scores) if recent_scores else 0
            win_rate = wins / total_games if total_games > 0 else 0

            print(
                f"  Episode {episode:>5}/{episodes} | "
                f"Avg Score: {avg_score:>+6.2f} | "
                f"Win Rate: {win_rate:.1%} | "
                f"ε: {discovery.epsilon:.3f} | "
                f"States: {len(discovery.play_q)}"
            )

    # Save model.
    discovery.save(save_path)
    print(f"\n  Model saved to: {save_path}")
    print(f"  Total episodes trained: {discovery.episodes_trained}")
    print(f"  Play states learned: {len(discovery.play_q)}")
    print(f"  Pass states learned: {len(discovery.pass_q)}")
    print(f"  Total Q-updates: {discovery.total_updates}")

    # Final stats.
    if scores_history:
        overall_avg = sum(scores_history) / len(scores_history)
        last_500 = scores_history[-500:]
        recent_avg = sum(last_500) / len(last_500) if last_500 else 0
        first_500 = scores_history[:500]
        early_avg = sum(first_500) / len(first_500) if first_500 else 0

        print(f"\n  Learning Progress:")
        print(f"    Early avg score (first 500 shotas):  {early_avg:>+.2f}")
        print(f"    Recent avg score (last 500 shotas):  {recent_avg:>+.2f}")
        print(f"    Overall avg score:                    {overall_avg:>+.2f}")
        print(f"    Improvement: {recent_avg - early_avg:>+.2f}")


def play_demo(model_path: str):
    """Play a demo game with a trained model and show results."""
    discovery = DiscoveryAgent(training=False)
    try:
        discovery.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model not found at {model_path}")
        return

    print(f"\nLoaded model: {model_path}")
    print(f"  Episodes trained: {discovery.episodes_trained}")
    print(f"  Play states: {len(discovery.play_q)}")

    random_agents = [RandomHeartsAgent() for _ in range(3)]
    agents = [discovery, random_agents[0], random_agents[1], random_agents[2]]

    print(f"\n{'='*50}")
    print(f"  Playing 100 games (Discovery Agent vs 3 Random)")
    print(f"{'='*50}\n")

    wins = 0
    total_score = 0

    for i in range(100):
        game = HeartsGame(agents=agents)
        result = game.play()
        total_score += result.final_scores[0]
        if result.winner_id == 0:
            wins += 1

    avg_score = total_score / 100
    print(f"  Results over 100 games:")
    print(f"    Win rate: {wins}%")
    print(f"    Average final score: {avg_score:+.1f}")
    print(f"    (Random baseline win rate: ~25%)")


def watch_game(model_path: str | None = None):
    """
    Watch one full game played trick by trick with full visibility.

    Shows:
    - Each player's hand at the start
    - Card passing
    - Each trick: who played what, who won
    - Scoring at end of each shota
    - Final results

    If model_path is provided, Player 0 uses the trained discovery agent.
    Otherwise all players are random.
    """
    from environments.hearts.player import HeartsPlayer
    from environments.hearts.environment import HeartsEnvironment
    from environments.hearts.observation import PassingObservation
    from environments.hearts.actions import PassCardsAction
    from environments.hearts.scoring import score_shota, count_penalties, QUEEN_OF_SPADES
    from environments.hearts.rules import trick_winner as tw_func
    from intelligence.core.cards.deck import Deck
    from intelligence.core.cards.suit import Suit

    PLAYER_NAMES = {0: "Discovery", 1: "Random-1", 2: "Random-2", 3: "Random-3"}
    SUIT_COLORS = {
        Suit.HEARTS: "\033[91m",    # Red
        Suit.DIAMONDS: "\033[91m",  # Red
        Suit.SPADES: "\033[97m",    # White
        Suit.CLUBS: "\033[97m",     # White
    }
    RESET = "\033[0m"

    def card_str(card):
        color = SUIT_COLORS.get(card.suit, "")
        return f"{color}{card.rank.symbol}{card.suit.symbol}{RESET}"

    def hand_str(hand):
        # Sort by suit then rank for readability.
        from intelligence.core.cards.rank import Rank as R
        suit_order = {Suit.SPADES: 0, Suit.HEARTS: 1, Suit.CLUBS: 2, Suit.DIAMONDS: 3}
        rank_order = {
            R.TWO: 2, R.THREE: 3, R.FOUR: 4, R.FIVE: 5, R.SIX: 6,
            R.SEVEN: 7, R.EIGHT: 8, R.NINE: 9, R.TEN: 10,
            R.JACK: 11, R.QUEEN: 12, R.KING: 13, R.ACE: 14,
        }
        sorted_hand = sorted(hand, key=lambda c: (suit_order.get(c.suit, 9), rank_order.get(c.rank, 0)))
        return " ".join(card_str(c) for c in sorted_hand)

    # Setup agents.
    if model_path:
        agent0 = DiscoveryAgent(training=False)
        try:
            agent0.load(model_path)
            PLAYER_NAMES[0] = f"Discovery({agent0.episodes_trained}ep)"
        except FileNotFoundError:
            print(f"Model not found at {model_path}, using random agent.")
            agent0 = RandomHeartsAgent()
            PLAYER_NAMES[0] = "Random-0"
    else:
        agent0 = RandomHeartsAgent()
        PLAYER_NAMES[0] = "Random-0"

    agents = [agent0, RandomHeartsAgent(), RandomHeartsAgent(), RandomHeartsAgent()]
    players = [HeartsPlayer(player_id=i) for i in range(4)]
    total_scores = {i: 0 for i in range(4)}
    dealer_id = 0

    print(f"\n{'='*65}")
    print(f"  SUDANESE HEARTS — Game Watch Mode")
    print(f"  Players: {', '.join(PLAYER_NAMES.values())}")
    print(f"{'='*65}")

    for shota_num in range(1, 6):
        print(f"\n{'─'*65}")
        print(f"  SHOTA {shota_num}/5   (Dealer: {PLAYER_NAMES[dealer_id]})")
        print(f"{'─'*65}")

        # Reset players.
        for p in players:
            p.reset_shota()

        # Deal.
        deck = Deck()
        deck.shuffle()
        for p in players:
            p.receive_cards(deck.deal(13))

        # Show hands after deal.
        print(f"\n  Hands after dealing:")
        for p in players:
            print(f"    {PLAYER_NAMES[p.player_id]:>15}: {hand_str(p.hand)}")

        # Passing phase.
        print(f"\n  Card Passing (each passes 4 cards to the left):")
        cards_to_pass = {}
        for p in players:
            obs = PassingObservation(player_id=p.player_id, hand=list(p.hand))
            action = agents[p.player_id].act(obs)
            cards_to_pass[p.player_id] = action.cards
            passed_str = " ".join(card_str(c) for c in action.cards)
            receiver = PLAYER_NAMES[(p.player_id + 1) % 4]
            print(f"    {PLAYER_NAMES[p.player_id]:>15} → {receiver}: {passed_str}")

        # Execute pass.
        for p in players:
            p.remove_cards(list(cards_to_pass[p.player_id]))
        for p in players:
            receiver_id = (p.player_id + 1) % 4
            players[receiver_id].receive_cards(list(cards_to_pass[p.player_id]))

        # Show hands after passing.
        print(f"\n  Hands after passing:")
        for p in players:
            print(f"    {PLAYER_NAMES[p.player_id]:>15}: {hand_str(p.hand)}")

        # Play 13 tricks.
        print(f"\n  Trick Play:")
        first_leader = (dealer_id + 1) % 4
        env = HeartsEnvironment(players, first_leader)

        trick_num = 1
        while not env.is_shota_complete():
            leader_id = env.current_trick.leading_player_id if env.current_trick else first_leader
            plays = []

            for i in range(4):
                current_pid = env.current_player_id()
                obs = env.observe(current_pid)
                action = agents[current_pid].act(obs)
                plays.append((current_pid, action.card))
                winner_id = env.apply_action(action)

            # Display the trick.
            plays_str = "  ".join(
                f"{PLAYER_NAMES[pid]}:{card_str(card)}" for pid, card in plays
            )
            # Determine penalty cards in this trick.
            trick_cards = [card for _, card in plays]
            penalties_in_trick = sum(1 for c in trick_cards if c.suit == Suit.HEARTS)
            has_queen = QUEEN_OF_SPADES in trick_cards
            penalty_note = ""
            if penalties_in_trick > 0 or has_queen:
                pts = penalties_in_trick + (7 if has_queen else 0)
                penalty_note = f"  💔-{pts}"

            print(f"    Trick {trick_num:>2}: {plays_str}")
            print(f"             → Winner: {PLAYER_NAMES[winner_id]}{penalty_note}")
            trick_num += 1

        # Scoring.
        collected = {p.player_id: list(p.collected_cards) for p in players}
        tricks_won = {p.player_id: p.tricks_won for p in players}
        scores = score_shota(collected, tricks_won)

        print(f"\n  Shota {shota_num} Results:")
        print(f"    {'Player':<20} {'Tricks':>7} {'Hearts':>7} {'Q♠':>5} {'Score':>7}")
        print(f"    {'─'*50}")
        for pid in range(4):
            hearts_count = sum(1 for c in collected[pid] if c.suit == Suit.HEARTS)
            has_q = QUEEN_OF_SPADES in collected[pid]
            q_str = "YES" if has_q else ""
            print(
                f"    {PLAYER_NAMES[pid]:<20} {tricks_won[pid]:>7} "
                f"{hearts_count:>7} {q_str:>5} {scores[pid]:>+7}"
            )

        # Check special scenarios.
        zero_trick_pids = [pid for pid in range(4) if tricks_won[pid] == 0]
        all_trick_pids = [pid for pid in range(4) if tricks_won[pid] == 13]
        if all_trick_pids:
            print(f"    🏆 ALL TRICKS to {PLAYER_NAMES[all_trick_pids[0]]}! (+18)")
        elif len(zero_trick_pids) == 1:
            print(f"    🏆 FULL GALLON by {PLAYER_NAMES[zero_trick_pids[0]]}! (+20)")
        elif len(zero_trick_pids) == 2:
            names = " & ".join(PLAYER_NAMES[pid] for pid in zero_trick_pids)
            print(f"    🏆 HALF GALLON by {names}! (+10 each)")

        # Accumulate.
        for pid, score in scores.items():
            total_scores[pid] += score

        # Show running totals.
        totals_str = "  ".join(
            f"{PLAYER_NAMES[pid]}:{total_scores[pid]:+d}" for pid in range(4)
        )
        print(f"\n    Running Totals: {totals_str}")

        # Rotate dealer.
        dealer_id = (dealer_id + 1) % 4

    # Final results.
    print(f"\n{'='*65}")
    print(f"  FINAL RESULTS")
    print(f"{'='*65}")
    sorted_players = sorted(range(4), key=lambda pid: total_scores[pid], reverse=True)
    for rank, pid in enumerate(sorted_players, 1):
        marker = " 👑" if rank == 1 else (" 💀" if rank == 4 else "")
        print(f"    {rank}. {PLAYER_NAMES[pid]:<20} {total_scores[pid]:>+5}{marker}")
    print()


def play_demo(model_path: str):
    """Play 100 games with a trained model and show win rate."""
    discovery = DiscoveryAgent(training=False)
    try:
        discovery.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model not found at {model_path}")
        return

    print(f"\nLoaded model: {model_path}")
    print(f"  Episodes trained: {discovery.episodes_trained}")
    print(f"  Play states: {len(discovery.play_q)}")

    random_agents = [RandomHeartsAgent() for _ in range(3)]
    agents = [discovery, random_agents[0], random_agents[1], random_agents[2]]

    print(f"\n{'='*50}")
    print(f"  Playing 100 games (Discovery Agent vs 3 Random)")
    print(f"{'='*50}\n")

    wins = 0
    total_score = 0

    for i in range(100):
        game = HeartsGame(agents=agents)
        result = game.play()
        total_score += result.final_scores[0]
        if result.winner_id == 0:
            wins += 1

    avg_score = total_score / 100
    print(f"  Results over 100 games:")
    print(f"    Win rate: {wins}%")
    print(f"    Average final score: {avg_score:+.1f}")
    print(f"    (Random baseline win rate: ~25%)")


def show_stats(model_path: str):
    """Show model statistics."""
    discovery = DiscoveryAgent(training=False)
    try:
        discovery.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model not found at {model_path}")
        return

    print(f"\n{'='*50}")
    print(f"  Model Statistics: {model_path}")
    print(f"{'='*50}")
    print(f"  Episodes trained:   {discovery.episodes_trained}")
    print(f"  Total Q-updates:    {discovery.total_updates}")
    print(f"  Play states:        {len(discovery.play_q)}")
    print(f"  Pass states:        {len(discovery.pass_q)}")

    # Show top play states by visit frequency.
    if discovery.play_q:
        print(f"\n  Top learned play preferences:")
        sorted_states = sorted(
            discovery.play_q.items(),
            key=lambda x: max(x[1].values()) - min(x[1].values()) if x[1] else 0,
            reverse=True,
        )
        for state, actions in sorted_states[:10]:
            best_action = max(actions, key=actions.get) if actions else "?"
            worst_action = min(actions, key=actions.get) if actions else "?"
            spread = max(actions.values()) - min(actions.values()) if actions else 0
            print(
                f"    State {state}: prefers '{best_action}' "
                f"(avoids '{worst_action}', spread={spread:.2f})"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Sudanese Hearts — Discovery Agent Training"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Train command.
    train_parser = subparsers.add_parser("train", help="Train the discovery agent")
    train_parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of games to train"
    )
    train_parser.add_argument(
        "--model", type=str, default=None, help="Load existing model to continue training"
    )
    train_parser.add_argument(
        "--save", type=str, default=None, help="Path to save the trained model"
    )

    # Watch command.
    watch_parser = subparsers.add_parser("watch", help="Watch one game trick by trick")
    watch_parser.add_argument(
        "--model", type=str, default=None, help="Path to trained model for Player 0"
    )

    # Play command (batch evaluation).
    play_parser = subparsers.add_parser("play", help="Play 100 games and show win rate")
    play_parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model"
    )

    # Stats command.
    stats_parser = subparsers.add_parser("stats", help="Show model statistics")
    stats_parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model"
    )

    args = parser.parse_args()

    if args.command == "train":
        train(episodes=args.episodes, model_path=args.model, save_path=args.save)
    elif args.command == "watch":
        watch_game(model_path=args.model)
    elif args.command == "play":
        play_demo(model_path=args.model)
    elif args.command == "stats":
        show_stats(model_path=args.model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
