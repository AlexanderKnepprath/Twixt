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
