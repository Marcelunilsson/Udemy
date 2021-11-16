import random


suits = ('Hearts', 'Diamonds', 'Spades', 'Clubs')
ranks = ('Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight',
         'Nine', 'Ten', 'Jack', 'Queen', 'King', 'Ace')
values = {'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5, 'Six': 6,
          'Seven': 7, 'Eight': 8, 'Nine': 9, 'Ten': 10, 'Jack': 10,
          'Queen': 10, 'King': 10, 'Ace': 11}


class card:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank
        self.value = values[rank]

    def __str__(self) -> str:
        return f"{self.rank} of {self.suit}"


class deck:
    def __init__(self):
        self.deck = [card(suit, rank) for rank in ranks for suit in suits]

    def __str__(self) -> str:
        return print("\n".join([str(card) for card in self.deck]))

    def shuffle(self):
        random.shuffle(self.deck)

    def deal(self):
        return self.deck.pop()


class player:
    def __init__(self, name, balance):
        self.name = name
        self.balance = balance
        self.hand = []

    def bet(self, amount):
        if amount > self.balance:
            print(f"Your balance is: {self.balance}")
        else:
            self.balance -= amount
            return amount

    def add_card_to_hand(self, card):
        self.hand.append(card)

    def show_hand(self):
        print("\n".join([str(card) for card in self.hand]))

    def reset_hand(self):
        self.hand = []


class blackjack_player(player):
    def total_value(self):
        val = sum([card.value for card in self.hand])
        for aces in range([card.rank for card in self.hand].count("Ace")):
            if val > 21:
                val -= 10


class blackjack_dealer(blackjack_player):

