from intelligence.core.cards.deck import Deck


def test_deck_has_52_cards():
    deck = Deck()

    assert len(deck) == 52


def test_deck_deals_cards():
    deck = Deck()

    hand = deck.deal(13)

    assert len(hand) == 13
    assert len(deck) == 39


def test_deck_cannot_deal_more_than_available():
    deck = Deck()

    try:
        deck.deal(53)
        assert False
    except ValueError:
        assert True