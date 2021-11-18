# %%
import numpy as np


class Primes:
    def __init__(self):
        self.primes = [2]
        self.eval_nbr = 3

    def is_prime(self):
        return (
            sum(
                (self.eval_nbr % n == 0)
                for n in self.primes
                if n <= np.sqrt(self.eval_nbr)
            )
            == 0
        )

    def print_prime(self):
        print(f"Prime number found: {self.eval_nbr}\nAll Primes found: {self.primes}")

    def next_prime(self):
        while not self.is_prime():
            self.eval_nbr += 1
        self.primes.append(self.eval_nbr)
        self.print_prime()
        self.eval_nbr += 1


def find_factorial(n):
    if n == 0:
        return 1
    else:
        return n * find_factorial(n - 1)


def collatz_conj(n):
    steps = 0
    while n > 1:
        steps += 1
        if n % 2 == 0:
            n /= 2
        else:
            n = n * 3 + 1
    return steps


# %%
primes = Primes()
# %%
primes.next_prime()
# %%
find_factorial(6)
# %%
collatz_conj(19)
# %%
