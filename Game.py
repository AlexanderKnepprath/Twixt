"""Game.py: Sets up and defines interactions with the game board"""

__author__      = "Alexander Knepprath"

# constants
BOARD_SIZE = 13 # 24 holes + 23 in between spots

# important definitions
board = list()

# Set up the default board
for i in range(BOARD_SIZE):
    board_row = list()
    for j in range(BOARD_SIZE):
        if (i == 0 or i == BOARD_SIZE-1) and (j == 0 or j == BOARD_SIZE-1): 
            board_row.append(None)
        elif i % 2 == 1 and j % 2 == 1:
            board_row.append(None)
        else:
            board_row.append(0)
    board.append(board_row)


def print_pegs():
    """ 
    This method prints the peg locations on the board
    """

    pegs = list()

    for i in range(len(board)): # we want the number value, not the actual board row, so we can do math

        peg_row = list()
        if i % 2 == 0: # only even rows will have any pegs
            for j in range(len(board[i])):
                if j % 2 == 0: # only even columns will have any pegs
                    peg_row.append(board[i][j]) # append the board value to the peg row
        
            print(peg_row)

    print(pegs)


def print_board():
    """ 
    This method prints the entire board
    """

    for i in board:
        print(i)


def add_peg(player: int, x: int, y: int):
    """ 
    Adds a peg to the board
    
    :player: The player who added the peg. An integer, either 1 for you or -1 for opponent.
    :x: The x coordinate (on the pegs board) to add the peg at. An integer between 0 and 23.
    :y: The y coordinate (on the pegs board) to add the peg at. An integer between 0 and 23.

    :return: True if the peg was added, False if the peg could not be added.
    """

    # Convert x and y values from peg coordinates to board coordinates
    trueX = x*2
    trueY = y*2

    print("Placing peg at true position (" + str(trueX) + "," + str(trueY) + ")")

    # Reject placement if space already occupied
    if board[trueY][trueX] != 0:
        return False
    
    #TODO: REJECT PLACEMENT IF BEHIND THE OPPONENT'S END LINE

    # Set peg slot to player's value (-1 or 1)
    board[trueY][trueX] = player

    # Check possible bridges and place if possible
    if check_bridge(trueX, trueY, trueX+2, trueY+4): # right 1 peg, down 2 pegs
        place_bridge(player, 2, trueX+1, trueY+2)   
    if check_bridge(trueX, trueY, trueX+4, trueY+2): # right 2 pegs, down 1 peg
        place_bridge(player, 2, trueX+2, trueY+1)   
    if check_bridge(trueX, trueY, trueX+4, trueY-2): # right 2 pegs, up 1 peg
        place_bridge(player, 1, trueX+2, trueY-1)   
    if check_bridge(trueX, trueY, trueX+2, trueY-4): # right 1 peg, up 2 pegs
        place_bridge(player, 1, trueX+1, trueY-2)   
    if check_bridge(trueX, trueY, trueX-2, trueY-4): # left 1 peg, up 2 pegs
        place_bridge(player, 2, trueX-1, trueY-2)   
    if check_bridge(trueX, trueY, trueX-4, trueY-2): # left 2 pegs, up 1 peg
        place_bridge(player, 2, trueX-2, trueY-1)   
    if check_bridge(trueX, trueY, trueX-4, trueY+2): # left 2 pegs, down 1 peg
        place_bridge(player, 1, trueX-2, trueY+1)   
    if check_bridge(trueX, trueY, trueX-2, trueY+4): # left 1 peg, down 2 pegs
        place_bridge(player, 1, trueX-1, trueY+2)   

    return True


def check_bridge(pos1x: int, pos1y: int, pos2x: int, pos2y: int):
    """
    Checks whether a bridge can be built.

    :pos: Position coordinates (on the full board). An integer between 0 and 46.

    :return: true if a bridge can be built in that location, false if not.
    """

    # check if the position values are illegal
    if not (pos1x % 2 == 0 and pos1y % 2 == 0 and pos2x % 2 == 0 and pos2y % 2 == 0):
        return False
    
    if pos1x < 0 or pos1x > BOARD_SIZE-1: 
        return False
    if pos1y < 0 or pos1y > BOARD_SIZE-1:
        return False
    if pos2x < 0 or pos2x > BOARD_SIZE-1:
        return False
    if pos2y < 0 or pos2y > BOARD_SIZE-1:
        return False
    
    print("Checking bridge between (" + str(pos1x) + "," + str(pos1y) + ") and (" + str(pos2x) + "," + str(pos2y) + ")")
    
    # check if pos1 and pos2 have the same color pegs
    pos1val = board[pos1y][pos1x]
    pos2val = board[pos2y][pos2x]
    if pos1val != pos2val: 
        return False
    
    # print("Found same color peg at (" + str(pos2x) + "," + str(pos2y) + ")!")
    
    # check if pos1 is a knight's move from pos2
    xDiff = pos2x - pos1x
    yDiff = pos2y - pos1y 

    if not ((abs(xDiff) == 2 and abs(yDiff) == 4) or (abs(xDiff) == 4 and abs(yDiff) == 2)): 
        return False
    
    # determine relevant bridge locations and check them for possible conflicts
    xDir = int(xDiff/abs(xDiff)) # unit direction of x2 relative to x1, either 1 or -1
    yDir = int(yDiff/abs(yDiff)) # unit direction of y2 relative to y1, either 1 or -1

    # for the direction with differential = 4, we want to check values at 1u, 2u, and 3u (u = unit direction)
    # for the direction with differential = 2, we want to check values at 0u, 1u, and 2u (u = unit direction)
    xRange = list()
    yRange = list()

    if abs(yDiff) == 4: # if the y direction has differential = 4
        yRange = range(pos1y + yDir, pos1y + yDiff, yDir) # set y-range to be pos1y + 1u to pos1y + 3u
        xRange = range(pos1x, pos1x + xDiff + xDir, xDir) # set x-range to be pos1x to pos1x + 2u
    elif abs(xDiff) == 4: # if the x direction has differential = 4    
        xRange = range(pos1x + xDir, pos1x + xDiff, xDir) # set x-range to be pos1y + 1u to pos1y + 3u
        yRange = range(pos1y, pos1y + yDiff + yDir, yDir) # set y-range to be pos1x to pos1x + 2u 

    print(xRange)
    print(yRange)

    for i in yRange:
        for j in xRange:
            if (i + j) % 2 == 1: # if the sum of x and y is odd, then we have a bridge to check
                if board[i][j] != 0: # if there is already a bridge here, we cannot make a bridge
                    return False
                
    # Given no reason why a bridge cannot be placed, return True!
    print("Bridge approved between (" + str(pos1x) + "," + str(pos1y) + ") and (" + str(pos2x) + "," + str(pos2y) + ") <<---")
    return True;
                
def place_bridge(player: int, direction: int, x: int, y: int):
    """
    Places a bridge at a particular bridge coordinate. 
    This method should not be used before using check_bridge().
    This method will attempt to sanitize inputs but cannot guarantee that a bridge placement is legal.

    :player: The player who owns this bridge
    :direction: A value for the slope of the bridge, either 1 for upwards or 2 for downwards
    :x: The x coordinate of the bridge
    :y: The y coordinate of the bridge

    :return: True if the bridge was placed, False if the bridge could not be placed. 
    """

    bridge_value = player * direction # combines the player value with the slope value of the bridge

    if x + y % 2 == 0: # Bridges can only be placed where the sum of the coordinates is odd.
        return False
    
    board[y][x] = bridge_value
    return True
            
            
add_peg(-1, 3, 2)
add_peg(-1, 5, 3)  
add_peg(1, 5, 2)
add_peg(1, 3, 3)
print_pegs()
print_board()