"""
    Implements the game of twixt with bridges stored as vectors in separate image matrices
"""

__author__ = "Alexander Knepprath"

import numpy as np

DEBUG_MODE = False

class TwixtEnvironment:

    """
        Creates a new Twixt board

        :param board_size: The size of the board, in pegs squared, including pegs beyond the end lines
    """
    def __init__(self, board_size):
        self.board_size = board_size
        self.reset()


    def reset(self):
        self.winner = None
        self.current_player = 1 # starting player
        self.board = np.zeros((9, self.board_size, self.board_size), np.int8)
    
    """
        Adds a peg to the pegboard and automatically adds any possible bridges

        :param player: An integer, either 1 for the player or -1 for the opponent
        :param location: A tuple of the form (x,y) for the coordinates of the peg

        :return: True if the peg is placed, false if it is not placed because location is illegal or occupied
    """
    def add_peg(self, position: tuple):

        # only add a peg if there is no winner
        if self.winner == None:

            player = self.current_player

            # verify that the position is valid
            if position[0] < 0 or position[0] >= self.board_size or position[1] < 0 or position[1] >= self.board_size:
                return False
            
            # verify that the peg location is not already occupied
            if not self.is_legal_move(player, position):
                return False
            
            # place peg
            self.board[0, position[0], position[1]] = player

            # place bridges at 8 candidate locations, if possible
            self.place_bridge(player, position, (position[0]+1, position[1]+2)) # up 2, right 1
            self.place_bridge(player, position, (position[0]+2, position[1]+1)) # up 1, right 2
            self.place_bridge(player, position, (position[0]+2, position[1]-1)) # down 1, right 2
            self.place_bridge(player, position, (position[0]+1, position[1]-2)) # down 2, right 1
            self.place_bridge(player, position, (position[0]-1, position[1]-2)) # down 2, left 1
            self.place_bridge(player, position, (position[0]-2, position[1]-1)) # down 1, left 2
            self.place_bridge(player, position, (position[0]-2, position[1]+1)) # up 1, left 2
            self.place_bridge(player, position, (position[0]-1, position[1]+2)) # up 2, left 1

            # check to see if this player has won the game
            if self.has_won(player):
                self.current_player = 0
                self.winner = player
                return True
            
            # check to see if the next player has a legal move
            if len(self.get_all_legal_moves(player * -1)) == 0:

                # if not, it's a draw.
                self.current_player = 0
                self.winner = 0
                return True
            
            # otherwise, set the current player to the next player and return true
            self.current_player = -self.current_player
            return True
        
        else:
            raise ValueError("The game has ended. Please reset the environment.")
        

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
        if not self.board[0, pos1[0], pos1[1]] == self.board[0, pos2[0], pos2[1]]:
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
            print_if_debug("XX - Conflict found! - XX")
            return False

        # if there is no conflict, we make a bridge in this spot by adjusting the vector images
        self.board[direction, leftpoint[0], leftpoint[1]] = player # update the leftpoint on the proper direction map (-1 b/c np arrays start at index 0)
        self.board[direction + 4, rightpoint[0], rightpoint[1]] = player # update the rightpoint on the proper direction map (-1 b/c np arrays start at index 0)

    """
        Returns true if there is a bridge at the given position, directed in the given direction

        :param position: A tuple, (x,y) coordinates of the position to check
        :param direction: An integer 1-8, for the direction of the given bridge

        :return: true if there is a bridge, false if there is no bridge
    """
    def bridge_at(self, position: tuple, direction: int):
        return abs(self.board[direction, position[0], position[1]]) == 1
    

    """
        Returns true if the player has won the game

        :param player: An integer, either 1 or -1, signifying the player to check for

        :return: True if and only if the player has made a complete bridge from one of their ends to the other
    """
    def has_won(self, player: int):
        print_if_debug("Checking for win")
        for i in range(self.board_size):
            if player == 1:
                if self.board[0, 0, i] == 1:
                    print_if_debug("Found player 1 peg beyond end line at (" + str(0) + ", " + str(i) + ")")
                if self.board[0, 0, i] == 1 and self.connects_to_end(1, (0, i), list()):
                    return True
            elif player == -1:
                if self.board[0, i, 0] == -1 and self.connects_to_end(-1, (i, 0), list()):
                    return True


    """
        Recursive method: returns true if the given peg connects to the end side (right side for p1, bottom for p2)

        :param player: An integer, the player to check the win condition for, either 1 for p1 or -1 for p2
        :param position: The peg to check
        :param checked_list: A list of all points that have been previously checked (to prevent infinite loops)
    """
    def connects_to_end(self, player: int, position: tuple, checked_list: list):
        print_if_debug("Connection made at " + str(position))

        # first, verify that the peg to check is actually controlled by the player
        if not self.board[0, position[0], position[1]] == player:
            print_if_debug("Peg at " + str(position) + " is not controlled by player " + str(player))
            return False

        # check if the peg is already beyond the end line
        if (player == 1 and position[0] == self.board_size - 1) or (player == -1 and position[1] == self.board_size - 1):
            print_if_debug("Peg at " + str(position) + " is beyond the opposite end line!")
            return True
        
        # if we haven't reached the end line, but the peg is controlled by the player, add the peg to the checked_list
        checked_list.append(position)

        # now check each vector map for a bridge
        for i in range(1, 9):

            # get the coordinates of the hypothetical new bridge endpoint
            new_position = (0, 0)
            if i == 1:
                new_position = (position[0]+1, position[1]+2)
            elif i == 2:
                new_position = (position[0]+2, position[1]+1)
            elif i == 3:
                new_position = (position[0]+2, position[1]-1)
            elif i == 4:
                new_position = (position[0]+1, position[1]-2)
            elif i == 5:
                new_position = (position[0]-1, position[1]-2)
            elif i == 6:
                new_position = (position[0]-2, position[1]-1)
            elif i == 7:
                new_position = (position[0]-2, position[1]+1)
            elif i == 8:
                new_position = (position[0]-1, position[1]+2)

            # check if there's actually a bridge there
            if self.board[i, position[0], position[1]] == player:
                print_if_debug("Found bridge to " + str(new_position) + " from " + str(position))

                # if so, make sure that we haven't already checked the new point
                new_position_checked = False
                for i in checked_list:
                    if new_position == i:
                        new_position_checked = True
                        print_if_debug("Already checked " + str(new_position) + " from " + str(position))

                # if the new point connects to the end, then we also connect to the end
                if (not new_position_checked) and self.connects_to_end(player, new_position, checked_list):
                    print_if_debug("Peg at " + str(position) + " connects to end via " + str(new_position))
                    return True
                    
        # nothing was found that connects to the end, so return false
        print_if_debug("Dead end at " + str(position))
        return False
    

    def is_legal_move(self, player: int, position: tuple):

        # if peg is beyond opponent's goal line
        if (position[0] == 0 or position[0] == self.board_size-1) and player == -1:
            return False
        elif (position[1] == 0 or position[1] == self.board_size-1) and player == 1:
            return False
        
        # if peg is already occupied
        elif self.board[0, position[0], position[1]] != 0:
            return False

        return True  
    

    def get_all_legal_moves(self, player:int):
        # Initialize an empty list to store legal moves
        legal_moves = []

        # Generate all legal moves (valid placements of pegs)
        for x in range(self.board_size):
            for y in range(self.board_size):
                # Check if the position (x, y) is empty and can be legally occupied by a peg
                if self.is_legal_move(player, (x, y)):  # Implement the function is_legal_move to check if the move is legal
                    # Add the coordinates of the legal move to the action space
                    legal_moves.append([x, y])

        return legal_moves
    

def print_if_debug(string:str):
    if (DEBUG_MODE):
        print(string)