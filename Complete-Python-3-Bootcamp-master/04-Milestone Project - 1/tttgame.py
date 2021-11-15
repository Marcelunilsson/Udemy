# %%
import numpy as np
from dataclasses import dataclass


@dataclass
class player:
    name: str
    marker: str
    score: int = 0


class TickTackToe:
    def __init__(self):
        self.p1 = player(input("Name of player one,(O)."), "[O]")
        self.p2 = player(input("Name of player two, (X)."), "[X]")
        self.grid = np.reshape(["[ ]"]*9, (3, 3))

    def show_grid(self):
        print(self.grid)

    def place_marker(self, player):
        available_moves = np.where(self.grid == "[ ]")
        print(f"Available moves: {available_moves}")
        print(f"Player: {player.name} \n Marker: {player.marker}")
        self.show_grid()
        row = int(input("What row to place marker on: "))
        col = int(input("What column to place marker on: "))
        self.grid[row, col] = player.marker

    def win_check(self, player):
        mark = player.marker
        if self.grid[:, ::-1].diagonal().tolist().count(mark) == 3 or self.grid.diagonal().tolist().count(mark) == 3:
            return True
        for i in range(3):
            if self.grid[i, :].tolist().count(mark) == 3 or self.grid[:, i].tolist().count(mark) == 3:
                return True
        return False

    def new_game(self):
        self.grid = np.reshape(["[ ]"]*9, (3, 3))
        p = self.p1
        while True:
            self.place_marker(p)
            if self.win_check(p):
                self.show_grid()
                print(f"Congratulations {p.name}, you have won!")
                break
            elif sum([row.count("[ ]") for row in self.grid.tolist()]) == 0:
                self.show_grid()
                print("Well played both of you, the game is an draw!")
                break
            else:
                if p == self.p1:
                    p = self.p2
                else:
                    p = self.p1


g = TickTackToe()
g.new_game()

# %%
