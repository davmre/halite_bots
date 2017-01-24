import hlt
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move, Square
from networking import *
import random

myID, game_map = hlt.get_init()
sendInit("MyPythonBot")

def assign_move(square):

    """
    for direction, neighbor in enumerate(game_map.neighbors(square)):
        if neighbor.owner != myID and neighbor.strength < square.strength:
            return Move(square, direction)

    if square.strength < 5 * square.production:
        return Move(square, STILL)
    else:
        return Move(square, random.choice((NORTH, WEST)))
    """
    return Move(square, NORTH)

while True:
    game_map.get_frame()
    moves = [assign_move(square) for square in game_map if square.owner == myID]
    hlt.send_frame(moves)