import twixt

def bridges_built(env, player, position):
    board = env.board
    counter = 0

    for i in range(1,9):
        if board[position[0]][position[1]][i] == player:
            if i % 4 == 0 or i % 4 == 3:
                counter += 2
            else:
                counter += 1

    return counter


def greatest_horizontal_distance_between_connected_pegs(env, player, position):

    far_left = position[0]
    far_right = position[0]

    checked_list = []

    far_left, far_right, _ = check_peg_and_connections(env, player, position, far_left, far_right, checked_list)

    return far_right - far_left



def check_peg_and_connections(env, player, position, far_left, far_right, checked_list):

    board = env.board

    # append the current peg to the checked list
    checked_list.append(position)
    
    # if the current position is more left than the current far left, set the new record
    if position[0] < far_left:
        far_left = position[0]

    # else, if the current position is more right than the current far right, set the new record
    elif position[0] > far_right:
        far_right = position[0]

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
        if board[position[0], position[1], i] == player:

            # if so, make sure that we haven't already checked the new point
            for i in checked_list:
                if new_position == i:
                    # if we have, move on
                    return far_left, far_right, checked_list
                
            # otherwise we check new position and its connections
            far_left, far_right, checked_list = check_peg_and_connections(env, player, new_position, far_left, far_right, checked_list)
    
    return far_left, far_right, checked_list
                


def num_cardinal_pegs(env, player, position):
    board = env.board
    x, y = position
    count = 0

    if x > 0 and board[x-1][y][0] == player:
        count += 1
    if x < 23 and board[x+1][y][0] == player:
        count += 1
    if y > 0 and board[x][y-1][0] == player:
        count += 1
    if y < 23 and board[x][y+1][0] == player:
        count += 1

    return count

    