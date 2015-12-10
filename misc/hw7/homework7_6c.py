#!/usr/bin/env python
from __future__ import division
import numpy as np
from utils import make_deck, get_card_counts
from time import time

#This code is modified from the original version by Henry Lin <halin2@illinois.edu>
__author__ = 'Jakub Klapacz <jklapac2@illinois.edu> && Henry Lin <halin2@illinois.edu>'

def make_deck(lands=24, spell1=10, spell2=10, spell3=10, spell4=2,
              spell5=5, spell6=6):
    """Creates a deck. The size of the deck varies depending on the
    number of each card. Lands are represented as 0. A spell of cost k is
    represented as k.

    Will raise a ValueError if the sum of the number of cards is
    not equal to 60.

    Returns
    -------
    A numpy array of size 60. The values of the cards are shuffled
    in this array.
    """
    decksize = lands + spell1 + spell2 + spell3 + spell4 + spell5 + spell6
    if decksize != 60:
        raise ValueError("The size of the deck does not equal 60.")

    # Using an absurd amount of operator overloading here
    deck = [0] * lands + [1] * spell1 + [2] * spell2 + [3] * spell3 \
           + [4] * spell4 + [5] * spell5 + [6] * spell6
    return np.random.permutation(np.array(deck))

def get_card_counts(hand):
    """Counts the number of occurances of each card. If a card
    has cost k, its count will appear in index k of the resulting
    array. In this case, the size of the return array is always 7,
    which the most expensive spell being cost 6.

    Remember that numpy arrays are zero indexed. Therefore, land
    cards (of cost zero) are placed in index zero, not 1.

    Returns
    -------
    A numpy array of length 7, corresponding to card counts.
    """

    # There's probably a numpy way of doing this without a
    # for loop.
    counts = np.zeros(7)
    for value in range(7):
        counts[value] = sum(hand == value)
    return counts

def play_round(lands_on_table, next_card_ind):
    """Plays the round. Notice that this function is a function inside
    a function, so it could modify the variables outside of its environment.
    This is called a closure.

    In this case, we're using the following outside variables:

    - deck
    - card_counts

    This is how I could have less input parameters than David's code.

    (Immutable objects in python, such as ints, cannot be access from
    outside the function. Unless you're working with python3; there is
    the "nonlocal" keyword.)

    **NOTE**: I was about to actually make this a non-closure, because
    having a closure is only going to cause confusion. Too late for that
    now...

    Returns
    -------
    played_spell : -1 if no spell was played. Otherwise, returns the value
    of the spell played.

    lands_on_table : The number of lands on the table
    """

    assert deck is not None, "Need to define a numpy array representing" \
                             "the deck, called 'deck'."
    assert card_counts is not None, "Need to define a numpy array representing" \
                                    "the counts of each card"

    next_card = deck[next_card_ind]
    card_counts[next_card] += 1

    # Put a land on the table if we have one in the hand
    if card_counts[0] > 0:
        lands_on_table += 1
        card_counts[0] -= 1

    # In this selection process, we are interested in picking the
    # most expensive card per round.
    for spell_value in range(1, lands_on_table + 1, 1):
        if card_counts[spell_value] > 0:
            card_counts[spell_value] -= 1
            return spell_value, lands_on_table
    return -1, lands_on_table

"""Part c of the homework in the textbook.

"Assume you have an MTG-DAF deck of

- 10 lands
- 10 spells of cost 1
- 10 spells of cost 2
- 10 spells of cost 3
- 2 spells of cost 4
- 2 spells of cost 5
- 16 spells of cost 6

It is properly shuffled, and you draw seven cards. At each turn,
you play a land if it is in your hand, and you always only play the
cheapest spell in your hand that you area able to play and you
never play two spells.

Part c: Extend your program to prepare a 6x6 table of spells. In the
t, s'th cell of the table, your program should place the probability
that the first spell you play is played on turn t, and has cost s. Notice
that many cells easily contain a zero.
"""

start = time()
n_simulations = 100000  # He calls these "inner simulations"

# 6 x 6 table of zeros. The first dimension (row) represents the
# turn, indexed at 0. The second dimension (col) represents the
# "spell value - 1". (For example, index 5 corresponds to a spell value of 6.)
counts = np.zeros((6, 6))

for sim in range(n_simulations):

    deck = make_deck(lands=10, spell1=10, spell2=10, spell3=10, spell4=2,
                     spell5=2, spell6=16)

    hand = deck[:7]
    card_counts = get_card_counts(hand)

    # From here, it's a matter of playing the game. Limit ourselves
    # to only 4 turns.
    n_turns = 6
    lands_on_table = 0
    next_card_ind = 7
    for turn_ind in range(n_turns):
        spell_value, lands_on_table \
            = play_round(lands_on_table, next_card_ind)
        next_card_ind += 1

        # TODO: Figure out whether a spell was played or not.
        # Record the spell played in the counts table above, in
        # counts[turn_ind][spell_value - 1]
        if spell_value > 0:
            spell_ind = spell_value - 1
            counts[turn_ind][spell_ind] += 1
            break

probabilities = counts / n_simulations
n_seconds = round(time() - start)

# The three lines below make a string table. You don't have to
# be able to understand it. (If you do, good job.)
probabilities = probabilities.tolist()
probabilities = map(lambda innerlist: '\t'.join(map(str, innerlist)), probabilities)
probabilities = "\n".join(probabilities)

print("Number of seconds: ".ljust(25) + str(format(n_seconds)))
print("Probabilities: ")
print(probabilities)
