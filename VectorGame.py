"""
    Implements the game of Twixt with bridges stored as vectors in separate image matrices
"""

__author__ = "Alexander Knepprath"

import numpy as np

# constants
BOARD_SIZE = 5

class Board:

    peg_matrix = np.zeros((BOARD_SIZE, BOARD_SIZE), np.int8)
    vector_matrix = np.zeros((8, BOARD_SIZE, BOARD_SIZE), np.int8)

    def __init__(self):
        True
    
    """
        Adds a peg to the pegboard and automatically adds any possible bridges

        :param player: An integer, either 1 for the player or -1 for the opponent
        :param location: A tuple of the form (x,y) for the coordinates of the peg

        :return: True if the peg is added, False if the peg cannot be added
    """
    def addPeg(self, player: int, position: tuple):
        
        # verify that the inputs are valid
        if not (player == -1 or player == 1):
            return False
        elif position[0] < 0 or position[0] >= BOARD_SIZE or position[1] < 0 or position[1] >= BOARD_SIZE:
            return False
        
        # verify that the peg location is not already occupied
        if self.peg_matrix[position[0], position[1]] != 0:
            return False
        
        # place peg
        self.peg_matrix[position[0], position[1]] = player

        # place bridges at 8 candidate locations, if possible
        self.place_bridge(player, position, (position[0]+1, position[1]+2)) # up 2, right 1
        self.place_bridge(player, position, (position[0]+2, position[1]+1)) # up 1, right 2
        self.place_bridge(player, position, (position[0]+2, position[1]-1)) # down 1, right 2
        self.place_bridge(player, position, (position[0]+1, position[1]-2)) # down 2, right 1
        self.place_bridge(player, position, (position[0]-1, position[1]-2)) # down 2, left 1
        self.place_bridge(player, position, (position[0]-2, position[1]-1)) # down 1, left 2
        self.place_bridge(player, position, (position[0]-2, position[1]+1)) # up 1, left 2
        self.place_bridge(player, position, (position[0]-1, position[1]+2)) # up 2, left 1

        

    """
        Attempts to place a bridge between two specified locations

        :param loc1: A tuple, the (x,y) coordinates of the first peg to connect
        :param loc2: A tuple, the (x,y) coordinates of the second peg to connect

        :return: True if the bridge was added, False if no bridge could be added
    """    
    def place_bridge(self, player, pos1: tuple, pos2: tuple):

        print("1 - Attempting to place bridge from (" + str(pos1[0]) + "," + str(pos1[1]) + ") to (" + str(pos2[0]) + "," + str(pos2[1]) + ")")

        # check if the position values are illegal    
        if pos1[0] < 0 or pos1[0] >= BOARD_SIZE: 
            return False
        if pos1[1] < 0 or pos1[1] >= BOARD_SIZE:
            return False
        if pos2[0] < 0 or pos2[0] >= BOARD_SIZE:
            return False
        if pos2[1] < 0 or pos2[1] >= BOARD_SIZE:
            return False
        
        print("2 - Legal position values!")
        
        # verify that the positions have the same colored pegs
        if not self.peg_matrix[pos1[0], pos1[1]] == self.peg_matrix[pos2[0], pos2[1]]:
            return False

        # verify that the positions are a knight's move apart
        xDiff = pos2[0] - pos1[0]
        yDiff = pos2[1] - pos1[1] 

        if not ((abs(xDiff) == 1 and abs(yDiff) == 2) or (abs(xDiff) == 2 and abs(yDiff) == 1)): 
            return False
        
        # check conflicting bridges
        
        # determine the left and right points, which will be the reference point (this means we don't need to check directions 5-8)
        leftpoint = (0,0)
        rightpoint = (0,0)
        if xDiff > 0:
            leftpoint = pos1
            rightpoint = pos2
        else:
            leftpoint = pos2
            rightpoint = pos1

        # determine the direction of the vector (integer 1-8, ordered clockwise starting with up 2 right 1)
        direction = 0
        slope = (rightpoint[1] - leftpoint[1])/(rightpoint[0] - leftpoint[0])
        if slope == 2:
            direction = 1
        elif slope == 0.5:
            direction = 2
        elif slope == -0.5:
            direction = 3
        elif slope == -2:
            direction = 4

        print("3 - Direction = " + str(direction))
        
        # iterate clockwise around points on the rectangle formed by leftpoint and rightpoint, checking for bridges which interfere
        conflict_found = False

        if direction == 1: # if direction is up 2 and right 1
            # first point (2, 3, 4)
            test_point = (leftpoint[0], leftpoint[1]+1) # up 1
            conflict_found = conflict_found or self.bridge_at(test_point, 2)
            conflict_found = conflict_found or self.bridge_at(test_point, 3)
            conflict_found = conflict_found or self.bridge_at(test_point, 4)
            # second point (3, 4)
            test_point = (leftpoint[0], leftpoint[1]+2) # up 2
            conflict_found = conflict_found or self.bridge_at(test_point, 3)
            conflict_found = conflict_found or self.bridge_at(test_point, 4)
            # third point (6, 7, 8)
            test_point = (leftpoint[0]+1, leftpoint[1]+1) # up 1, right 1
            conflict_found = conflict_found or self.bridge_at(test_point, 6)
            conflict_found = conflict_found or self.bridge_at(test_point, 7)
            conflict_found = conflict_found or self.bridge_at(test_point, 8)
            # fourth point (7, 8)
            test_point = (leftpoint[0]+1, leftpoint[1]) # right 1
            conflict_found = conflict_found or self.bridge_at(test_point, 7)
            conflict_found = conflict_found or self.bridge_at(test_point, 8)
        
        elif direction == 2: # if direction is up 1 and right 2
            # first point (3, 4)
            conflict_found = conflict_found or self.bridge_at((leftpoint[0], leftpoint[1]+1), 3)
            conflict_found = conflict_found or self.bridge_at((leftpoint[0], leftpoint[1]+1), 4)
            # second point (3, 4, 5)
            conflict_found = conflict_found or self.bridge_at((leftpoint[0]+1, leftpoint[1]+1), 3)
            conflict_found = conflict_found or self.bridge_at((leftpoint[0]+1, leftpoint[1]+1), 4)
            conflict_found = conflict_found or self.bridge_at((leftpoint[0]+1, leftpoint[1]+1), 5)
            # third point (7, 8)
            conflict_found = conflict_found or self.bridge_at((leftpoint[0]+2, leftpoint[1]), 7)
            conflict_found = conflict_found or self.bridge_at((leftpoint[0]+2, leftpoint[1]), 8)
            # fourth point (7, 8, 1)
            conflict_found = conflict_found or self.bridge_at((leftpoint[0]+1, leftpoint[1]), 7)
            conflict_found = conflict_found or self.bridge_at((leftpoint[0]+1, leftpoint[1]), 8)
            conflict_found = conflict_found or self.bridge_at((leftpoint[0]+1, leftpoint[1]), 1)

        elif direction == 3: # if direction is down 1 and right 2
            # first point (4, 5, 6)
            conflict_found = conflict_found or self.bridge_at((leftpoint[0]+1, leftpoint[1]), 4)
            conflict_found = conflict_found or self.bridge_at((leftpoint[0]+1, leftpoint[1]), 5)
            conflict_found = conflict_found or self.bridge_at((leftpoint[0]+1, leftpoint[1]), 6)
            # second point (5, 6)
            conflict_found = conflict_found or self.bridge_at((leftpoint[0]+2, leftpoint[1]), 5)
            conflict_found = conflict_found or self.bridge_at((leftpoint[0]+2, leftpoint[1]), 6)
            # third point (8, 1, 2)
            conflict_found = conflict_found or self.bridge_at((leftpoint[0]+1, leftpoint[1]-1), 8)
            conflict_found = conflict_found or self.bridge_at((leftpoint[0]+1, leftpoint[1]-1), 1)
            conflict_found = conflict_found or self.bridge_at((leftpoint[0]+1, leftpoint[1]-1), 2)
            # fourth point (1, 2)
            conflict_found = conflict_found or self.bridge_at((leftpoint[0], leftpoint[1]-1), 1)
            conflict_found = conflict_found or self.bridge_at((leftpoint[0], leftpoint[1]-1), 2)

        elif direction == 4: # if direction is down 2 and right 1
            # first point (5, 6)
            test_point = (leftpoint[0] + 1, leftpoint[1])
            conflict_found = conflict_found or self.bridge_at(test_point, 5)
            conflict_found = conflict_found or self.bridge_at(test_point, 6)
            # second point (5, 6, 7)
            conflict_found = conflict_found or self.bridge_at((leftpoint[0]+1, leftpoint[1]-1), 5)
            conflict_found = conflict_found or self.bridge_at((leftpoint[0]+1, leftpoint[1]-1), 6)
            conflict_found = conflict_found or self.bridge_at((leftpoint[0]+1, leftpoint[1]-1), 7)
            # third point (1, 2)
            conflict_found = conflict_found or self.bridge_at((leftpoint[0], leftpoint[1]-2), 1)
            conflict_found = conflict_found or self.bridge_at((leftpoint[0], leftpoint[1]-2), 2)
            # fourth point (1, 2, 3)
            conflict_found = conflict_found or self.bridge_at((leftpoint[0], leftpoint[1]-1), 1)
            conflict_found = conflict_found or self.bridge_at((leftpoint[0], leftpoint[1]-1), 2)
            conflict_found = conflict_found or self.bridge_at((leftpoint[0], leftpoint[1]-1), 3)

        if conflict_found:
            print("XX - Conflict found! - XX")
            return False

        # if there is no conflict, we make a bridge in this spot by adjusting the vector images
        print("4 - Making a bridge! Direction = " + str(direction) + " from " + str(leftpoint) + " to " + str(rightpoint))
        self.vector_matrix[direction - 1, leftpoint[0], leftpoint[1]] = player # update the leftpoint on the proper direction map (-1 b/c np arrays start at index 0)
        self.vector_matrix[direction + 3, rightpoint[0], rightpoint[1]] = player # update the rightpoint on the proper direction map (-1 b/c np arrays start at index 0)

    """
        Returns true if there is a bridge at the given position, directed in the given direction

        :param position: A tuple, (x,y) coordinates of the position to check
        :param direction: An integer 1-8, for the direction of the given bridge

        :return: true if there is a bridge, false if there is no bridge
    """
    def bridge_at(self, position: tuple, direction: int):
        return abs(self.vector_matrix[direction - 1, position[0], position[1]]) == 1

board = Board()
board.addPeg(1, (1, 1))
board.addPeg(1, (2, 3))
print(board.peg_matrix)
print(board.vector_matrix)
