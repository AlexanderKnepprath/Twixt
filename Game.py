"""Game.py: Sets up and defines interactions with the game board"""

__author__      = "Alexander Knepprath"

from easygraphics import * # graphics engine, probably for testing, may remove later

# constants
NUM_PEGS_PER_AXIS = 24
BOARD_SIZE = 47 #(NUM_PEGS_PER_AXIS * 2) - 1
GRAPHIC_SIZE = 18
PLAYER_ONE_COLOR = Color.RED
PLAYER_ONE_LIGHT_COLOR = rgb(255, 230, 230)
PLAYER_TWO_COLOR = Color.BLUE
PLAYER_TWO_LIGHT_COLOR = rgb(230, 230, 255)
EMPTY_PEG_COLOR = Color.DARK_GRAY
INVALID_PEG_COLOR = Color.LIGHT_GRAY
GUIDE_LINES_COLOR = Color.BLACK

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

    for i in range(0, len(board), 2): # we want even number values
        # create a 1-D array for pegs
        peg_row = list()

        for j in range(0, len(board[i]), 2): # only even columns will have any pegs
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

    pos1 = [pos1x, pos1y]
    pos2 = [pos2x, pos2y]

    # check if the position values are illegal
    if not (pos1[0] % 2 == 0 and pos1[1] % 2 == 0 and pos2[0] % 2 == 0 and pos2[1] % 2 == 0):
        return False
    
    if pos1[0] < 0 or pos1[0] > BOARD_SIZE-1: 
        return False
    if pos1[1] < 0 or pos1[1] > BOARD_SIZE-1:
        return False
    if pos2[0] < 0 or pos2[0] > BOARD_SIZE-1:
        return False
    if pos2[1] < 0 or pos2[1] > BOARD_SIZE-1:
        return False
    
    print("Checking bridge between (" + str(pos1[0]) + "," + str(pos1[1]) + ") and (" + str(pos2[0]) + "," + str(pos2[1]) + ")")
    
    # check if pos1 and pos2 have the same color pegs
    pos1val = board[pos1[1]][pos1[0]]
    pos2val = board[pos2[1]][pos2[0]]
    if pos1val != pos2val: 
        return False
    
    # check if pos1 is a knight's move from pos2
    xDiff = pos2[0] - pos1[0]
    yDiff = pos2[1] - pos1[1] 

    if not ((abs(xDiff) == 2 and abs(yDiff) == 4) or (abs(xDiff) == 4 and abs(yDiff) == 2)): 
        return False

    # determine whether new bridge will have positive (1) or negative (2) slope
    slope = 0
    if xDiff * yDiff < 0:
        slope = 1
    else:
        slope = 2
    
    # determine relevant bridge locations and check them for possible conflicts
    xDir = int(xDiff/abs(xDiff)) # unit direction of x2 relative to x1, either 1 or -1
    yDir = int(yDiff/abs(yDiff)) # unit direction of y2 relative to y1, either 1 or -1

    # for the direction with differential = 4, we want to check values at 1u, 2u, and 3u (u = unit direction)
    # for the direction with differential = 2, we want to check values at 0u, 1u, and 2u (u = unit direction)
    xRange = list()
    yRange = list()

    # check for bridges at the boundaries of the points (edge case)

    if abs(yDiff) == 4: # if the y direction has differential = 4
        # if there is a bridge at the near point and it doesn't match the slope of the new one, return false
        if board[pos1[1]][pos1[0] + xDir] != 0 and abs(board[pos1[1]][pos1[0] + xDir]) != slope: 
            return False
        # if there is a bridge at the far point and it doesn't match the slope of the new one, return false
        if board[pos1[1] + yDiff][pos1[0] + xDir] != 0 and abs(board[pos1[1] + yDiff][pos1[0] + xDir]) != slope:  
            return False  
 
    elif abs(xDiff) == 4: # if the x direction has differential = 4
        # if there is a bridge at the near point and it doesn't match the slope of the new one, return false
        if board[pos1[1] + yDir][pos1[0]] != 0 and abs(board[pos1[1] + yDir][pos1[0]]) != slope: 
            return False
        # if there is a bridge at the far point and it doesn't match the slope of the new one, return false
        if board[pos1[1] + yDir][pos1[0] + xDiff] != 0 and abs(board[pos1[1] + yDir][pos1[0] + xDiff]) != slope:  
            return False  


    # check for bridges between the points

    if abs(yDiff) == 4: # if the y direction has differential = 4
        yRange = range(pos1[1] + yDir, pos1[1] + yDiff, yDir) # set y-range to be pos1y + 1u to pos1y + 3u
        xRange = range(pos1[0], pos1[0] + xDiff + xDir, xDir) # set x-range to be pos1x to pos1x + 2u
    elif abs(xDiff) == 4: # if the x direction has differential = 4    
        xRange = range(pos1[0] + xDir, pos1[0] + xDiff, xDir) # set x-range to be pos1x + 1u to pos1x + 3u
        yRange = range(pos1[1], pos1[1] + yDiff + yDir, yDir) # set y-range to be pos1y to pos1y + 2u 

    for i in yRange:
        for j in xRange:
            if (i + j) % 2 == 1: # if the sum of x and y is odd, then we have a bridge location to check
                if board[i][j] != 0: # if there is already a bridge here, we need to check direction

                    print("Checking bridge at (" + str(j) + "," + str(i) + ")")
                    
                    # first, we need to get the x and y coordinates of the potential conflicting bridge
                    pc1 = [0, 0] # (x,y)
                    pc2 = [0, 0] # (x,y)

                    if i % 2 == 0: # this will be a primary vertical bridge
                        if abs(board[i][j]) == 1: # positive slope
                            pc1 = [j+1, i-2]
                            pc2 = [j-1, i+2]
                        elif abs(board[i][j]) == 2: # a negative slope
                            pc1 = [j+1, i+2]
                            pc2 = [j-1, i-2]
                    elif j % 2 == 0: # this will be a primary horizontal bridge
                        if abs(board[i][j]) == 1: # positive slope
                            pc1 = [j+2, i-1]
                            pc2 = [j-2, i+1]
                        elif abs(board[i][j]) == 2: # a negative slope
                            pc1 = [j+2, i+1]
                            pc2 = [j-2, i-1]

                    print(pos1)
                    print(pos2)
                    print(pc1)
                    print(pc2)

                    # now, if the new bridge shares no coordinates with the conflicting bridge, we cannot place
                    if not (pos1 == pc1 or pos1 == pc2 or pos2 == pc1 or pos2 == pc2):
                        print("A")
                        return False
                    
                    # assuming they share at least one peg, they must have the same slope
                    if abs(board[i][j]) != slope: 
                        print("B")
                        return False
                
    # Given no reason why a bridge cannot be placed, return True!
    print("Bridge approved between (" + str(pos1[0]) + "," + str(pos1[1]) + ") and (" + str(pos2[0]) + "," + str(pos2[1]) + ") <<---")
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


def add_pegs():
    add_peg(-1, 2, 3)
    add_peg(-1, 3, 5)   
    add_peg(1, 2, 5)
    add_peg(1, 3, 2)
    add_peg(-1, 7, 5)
    add_peg(-1, 8, 7)   
    add_peg(1, 3, 7)
    add_peg(1, 7, 3) 
    add_peg(-1, 5, 4) 
    # add_peg(1, 1, 3) 


# primary program loop
def mainloop():
    game_active = True   
    turn = 1

    while game_active:

        ###
        # run graphics engine
        ###      

        if turn == 1:
            set_background_color(PLAYER_ONE_LIGHT_COLOR)
            clear_device()
            print("Background Red")

        if turn == -1:
            set_background_color(PLAYER_TWO_LIGHT_COLOR) 
            clear_device()
            print("Background Blue") 

        # build the board lines
        # guide lines
        set_line_width(1)
        set_color(GUIDE_LINES_COLOR)
        draw_line(GRAPHIC_SIZE, GRAPHIC_SIZE*(BOARD_SIZE + 1)/2, GRAPHIC_SIZE*BOARD_SIZE, GRAPHIC_SIZE*(BOARD_SIZE + 1)/2)
        draw_line(GRAPHIC_SIZE*(BOARD_SIZE + 1)/2, GRAPHIC_SIZE, GRAPHIC_SIZE*(BOARD_SIZE + 1)/2, GRAPHIC_SIZE*BOARD_SIZE)
        set_line_width(3)
        # red lines
        set_color(PLAYER_ONE_COLOR)
        draw_line(GRAPHIC_SIZE*2, GRAPHIC_SIZE*2, GRAPHIC_SIZE*2, GRAPHIC_SIZE*(BOARD_SIZE - 1)) # top left to bottom left
        draw_line(GRAPHIC_SIZE*(BOARD_SIZE - 1), GRAPHIC_SIZE*2, GRAPHIC_SIZE*(BOARD_SIZE - 1), GRAPHIC_SIZE*(BOARD_SIZE - 1)) # top right to bottom right
        # blue lines
        set_color(PLAYER_TWO_COLOR)
        draw_line(GRAPHIC_SIZE*2, GRAPHIC_SIZE*2, GRAPHIC_SIZE*(BOARD_SIZE - 1), GRAPHIC_SIZE*2) # top left to top right
        draw_line(GRAPHIC_SIZE*2, GRAPHIC_SIZE*(BOARD_SIZE - 1), GRAPHIC_SIZE*(BOARD_SIZE - 1), GRAPHIC_SIZE*(BOARD_SIZE - 1)) # bottom left to bottom right
        # reset line width
        set_line_width(1)

        # loop through all locations
        for i in range(0, len(board)):
            for j in range(0, len(board[i])):
                    
                # get the value of this board location
                spot_value = board[i][j]

                # if the location is a peg location
                if i % 2 == 0 and j % 2 == 0:

                    # determine color based on value
                    set_color(Color.BLACK)
                    if spot_value == None:
                        set_color(INVALID_PEG_COLOR)
                        set_fill_color(INVALID_PEG_COLOR)
                    elif spot_value == 0:
                        set_fill_color(EMPTY_PEG_COLOR)
                    elif spot_value > 0:
                        set_fill_color(PLAYER_ONE_COLOR)
                    elif spot_value < 0:
                        set_fill_color(PLAYER_TWO_COLOR)

                    # print peg
                    draw_ellipse(GRAPHIC_SIZE*(j+1), GRAPHIC_SIZE*(i+1), GRAPHIC_SIZE/2, GRAPHIC_SIZE/2)
                    
                # if the location is a bridge location
                if (i + j) % 2 == 1 and spot_value != 0:
                        
                    # determine color based on value
                    set_color(Color.WHITE)
                    set_fill_color(Color.WHITE)
                    if spot_value < 0:
                        set_color(Color.BLUE)
                        set_fill_color(Color.BLUE)
                    elif spot_value > 0:
                        set_color(Color.RED)
                        set_fill_color(Color.RED)

                    # determine direction based on position and value
                    x1 = 0
                    y1 = 0
                    x2 = 0
                    y2 = 0

                    # note: in each case, the entire graph is offset by GRAPHIC_SIZE
                    if i % 2 == 0: # this will be a primarily vertical bridge
                        if abs(spot_value) == 1: # this will have positive slope
                            x1 = GRAPHIC_SIZE * (j+1) + GRAPHIC_SIZE
                            x2 = GRAPHIC_SIZE * (j-1) + GRAPHIC_SIZE
                            y1 = GRAPHIC_SIZE * (i-2) + GRAPHIC_SIZE
                            y2 = GRAPHIC_SIZE * (i+2) + GRAPHIC_SIZE
                        elif abs(spot_value) == 2: # this will have negative slope
                            x1 = GRAPHIC_SIZE * (j+1) + GRAPHIC_SIZE
                            x2 = GRAPHIC_SIZE * (j-1) + GRAPHIC_SIZE
                            y1 = GRAPHIC_SIZE * (i+2) + GRAPHIC_SIZE
                            y2 = GRAPHIC_SIZE * (i-2) + GRAPHIC_SIZE
                    elif j % 2 == 0: # this will be a primarily horizontal bridge
                        if abs(spot_value) == 1: # this will have positive slope
                            x1 = GRAPHIC_SIZE * (j+2) + GRAPHIC_SIZE
                            x2 = GRAPHIC_SIZE * (j-2) + GRAPHIC_SIZE
                            y1 = GRAPHIC_SIZE * (i-1) + GRAPHIC_SIZE
                            y2 = GRAPHIC_SIZE * (i+1) + GRAPHIC_SIZE
                        elif abs(spot_value) == 2: # this will have negative slope
                            x1 = GRAPHIC_SIZE * (j+2) + GRAPHIC_SIZE
                            x2 = GRAPHIC_SIZE * (j-2) + GRAPHIC_SIZE
                            y1 = GRAPHIC_SIZE * (i+1) + GRAPHIC_SIZE
                            y2 = GRAPHIC_SIZE * (i-1) + GRAPHIC_SIZE

                    # draw bridge
                    set_line_width(5)
                    draw_line(x1, y1, x2, y2)
                    set_line_width(1)

        
        ###
        # Player turns
        ###

        valid_peg = False
        while not valid_peg:
            # wait for a mouse click
            mouse = get_click()

            # get the coordinates of the mouse click
            pegX = int((mouse.x)/(2*GRAPHIC_SIZE))
            pegY = int((mouse.y)/(2*GRAPHIC_SIZE))

            # attempt to add peg and, if successful, break the while loop
            valid_peg = add_peg(turn, pegX, pegY)

        # change the turn
        if turn > 0:
            turn = -1
        else:
            turn = 1


# main function
def main():
    # add_pegs()
    # print_pegs()
    # print_board()

    init_graph((BOARD_SIZE+1) * GRAPHIC_SIZE, (BOARD_SIZE+1) * GRAPHIC_SIZE)
    set_render_mode(RenderMode.RENDER_AUTO)
    mainloop()
    close_graph()
  
  
# Using the special variable 
# __name__
if __name__=="__main__":
    main()