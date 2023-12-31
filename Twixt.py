"""
    Implements the game of twixt with bridges stored as vectors in separate image matrices
"""

__author__ = "Alexander Knepprath"

import numpy as np

class Board:

    """
        Creates a new Twixt board

        :param board_size: The size of the board, in pegs squared, including pegs beyond the end lines
    """
    def __init__(self, board_size):
        self.board_size = board_size
        self.board_matrix = np.zeros((9, self.board_size, self.board_size), np.int8)

    
    """
        Adds a peg to the pegboard and automatically adds any possible bridges

        :param player: An integer, either 1 for the player or -1 for the opponent
        :param location: A tuple of the form (x,y) for the coordinates of the peg

        :return: A tuple boolean.
        :return[0]: True if and only if the peg was added
        :return[1]: True if and only if the player has won the game
    """
    def add_peg(self, player: int, position: tuple):
        
        # verify that the inputs are valid
        if not (player == -1 or player == 1):
            return (False, False)
        elif position[0] < 0 or position[0] >= self.board_size or position[1] < 0 or position[1] >= self.board_size:
            return (False, False)
        
        # verify that the peg location is not already occupied
        if self.board_matrix[0, position[0], position[1]] != 0:
            return (False, False)
        
        # place peg
        self.board_matrix[0, position[0], position[1]] = player

        # place bridges at 8 candidate locations, if possible
        self.place_bridge(player, position, (position[0]+1, position[1]+2)) # up 2, right 1
        self.place_bridge(player, position, (position[0]+2, position[1]+1)) # up 1, right 2
        self.place_bridge(player, position, (position[0]+2, position[1]-1)) # down 1, right 2
        self.place_bridge(player, position, (position[0]+1, position[1]-2)) # down 2, right 1
        self.place_bridge(player, position, (position[0]-1, position[1]-2)) # down 2, left 1
        self.place_bridge(player, position, (position[0]-2, position[1]-1)) # down 1, left 2
        self.place_bridge(player, position, (position[0]-2, position[1]+1)) # up 1, left 2
        self.place_bridge(player, position, (position[0]-1, position[1]+2)) # up 2, left 1

        if self.has_won(player):
            return(True, True)

        return (True, False)
        

    """
        Attempts to place a bridge between two specified locations

        :param loc1: A tuple, the (x,y) coordinates of the first peg to connect
        :param loc2: A tuple, the (x,y) coordinates of the second peg to connect

        :return: True if the bridge was added, False if no bridge could be added
    """    
    def place_bridge(self, player, pos1: tuple, pos2: tuple):

        # check if the position values are illegal    
        if pos1[0] < 0 or pos1[0] >= self.board_size: 
            return False
        if pos1[1] < 0 or pos1[1] >= self.board_size:
            return False
        if pos2[0] < 0 or pos2[0] >= self.board_size:
            return False
        if pos2[1] < 0 or pos2[1] >= self.board_size:
            return False
        
        # verify that the positions have the same colored pegs
        if not self.board_matrix[0, pos1[0], pos1[1]] == self.board_matrix[0, pos2[0], pos2[1]]:
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
        self.board_matrix[direction, leftpoint[0], leftpoint[1]] = player # update the leftpoint on the proper direction map (-1 b/c np arrays start at index 0)
        self.board_matrix[direction + 4, rightpoint[0], rightpoint[1]] = player # update the rightpoint on the proper direction map (-1 b/c np arrays start at index 0)

    """
        Returns true if there is a bridge at the given position, directed in the given direction

        :param position: A tuple, (x,y) coordinates of the position to check
        :param direction: An integer 1-8, for the direction of the given bridge

        :return: true if there is a bridge, false if there is no bridge
    """
    def bridge_at(self, position: tuple, direction: int):
        return abs(self.board_matrix[direction, position[0], position[1]]) == 1
    

    """
        Returns true if the player has won the game

        :param player: An integer, either 1 or -1, signifying the player to check for

        :return: True if and only if the player has made a complete bridge from one of their ends to the other
    """
    def has_won(self, player: int):
        for i in range(self.board_size):
            if player == 1:
                if self.board_matrix[0, 0, i] == 1 and self.connects_to_end(1, (0, i), list()):
                    return True
            elif player == 2:
                if self.board_matrix[0, i, 0] == -1 and self.connects_to_end(-1, (i, 0), list()):
                    return True


    """
        Recursive method: returns true if the given peg connects to the end side (right side for p1, bottom for p2)

        :param player: An integer, the player to check the win condition for, either 1 for p1 or -1 for p2
        :param position: The peg to check
        :param checked_list: A list of all points that have been previously checked (to prevent infinite loops)
    """
    def connects_to_end(self, player: int, position: tuple, checked_list: list):
        
        # first, verify that the peg to check is actually controlled by the player
        if not self.board_matrix[0, position[0], position[1]] == player:
            return False

        # check if the peg is already beyond the end line
        if (player == 1 and position[0] == self.board_size - 1) or (player == -1 and position[1] == self.board_size - 1):
            return True
        
        # if we haven't reached the end line, but the peg is controlled by the player, add the peg to the checked_list
        checked_list.append(position)

        # now check each vector map for a bridge
        for i in range(1, 9):

            # get the coordinates of the hypothetical new bridge endpoint
            new_position = (0,0)
            if i == 0:
                new_position = (position[0]+1, position[1]+2)
            elif i == 1:
                new_position = (position[0]+2, position[1]+1)
            elif i == 2:
                new_position = (position[0]+2, position[1]-1)
            elif i == 3:
                new_position = (position[0]+1, position[1]-2)
            elif i == 4:
                new_position = (position[0]-1, position[1]-2)
            elif i == 5:
                new_position = (position[0]-2, position[1]-1)
            elif i == 6:
                new_position = (position[0]-2, position[1]+1)
            elif i == 7:
                new_position = (position[0]-1, position[1]+2)

            # check if there's actually a bridge there
            if self.board_matrix[i, position[0], position[1]] == 1:

                # if so, make sure that we haven't already checked the new point
                new_position_checked = False
                for i in checked_list:
                    if new_position == i:
                        new_position_checked = True

                # if the new point connects to the end, then we also connect to the end
                if (not new_position_checked) and self.connects_to_end(player, new_position, checked_list):
                    return True
                    
        # nothing was found that connects to the end, so return false
        return False
        



