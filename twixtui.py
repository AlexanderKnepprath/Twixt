"""
    Implements a User Interface for the Twixt board game
"""

__author__ = "Alexander Knepprath"

import twixt
import numpy as np
from easygraphics import *

# constants
GRAPHIC_SIZE = 36 # number of pixels per peg

GUIDE_LINE_COLOR = Color.LIGHT_GRAY # color for board center lines
EMPTY_PEG_COLOR = Color.DARK_GRAY
PEG_OUTLINE_COLOR = Color.BLACK
PLAYER_ONE_COLOR = Color.RED # peg and bridge color for p1
PLAYER_ONE_COLOR_LIGHT = rgb(255,245,245) # background color for p1
PLAYER_TWO_COLOR = Color.BLUE # peg and bridge color for p2
PLAYER_TWO_COLOR_LIGHT = rgb(245,245,255) # background color for p2

# Heatmap Colors
HEAT_COLOR = [rgb(255,255,245), 
              rgb(255,249,120), 
              rgb(255,243,59), 
              rgb(253,199,12),
              rgb(243,144,63),
              rgb(237,104,60),
              rgb(233,62,58),
              rgb(150,40,125)]

# init board
environment = twixt.TwixtEnvironment(24)

def renderEnvironment(env, heatmap):
    # set background color based on player turn
    set_line_width(1)    
    
    if not heatmap:
        if env.current_player == 1:
            set_background_color(PLAYER_ONE_COLOR_LIGHT)
            clear_device()
        elif env.current_player == -1:
            set_background_color(PLAYER_TWO_COLOR_LIGHT) 
            clear_device()      

    # draw board lines
    set_line_width(1)
    set_color(GUIDE_LINE_COLOR)
    draw_line(GRAPHIC_SIZE / 2, env.board_size * GRAPHIC_SIZE / 2, env.board_size * GRAPHIC_SIZE - GRAPHIC_SIZE/2, env.board_size * GRAPHIC_SIZE / 2)
    draw_line(env.board_size * GRAPHIC_SIZE / 2, GRAPHIC_SIZE / 2, env.board_size * GRAPHIC_SIZE / 2, env.board_size * GRAPHIC_SIZE - GRAPHIC_SIZE/2)
    set_line_width(3)
    set_color(PLAYER_ONE_COLOR)
    draw_line(GRAPHIC_SIZE, GRAPHIC_SIZE, GRAPHIC_SIZE, GRAPHIC_SIZE * (env.board_size - 1))
    draw_line(GRAPHIC_SIZE * (env.board_size - 1), GRAPHIC_SIZE, GRAPHIC_SIZE * (env.board_size - 1), GRAPHIC_SIZE * (env.board_size - 1))
    set_color(PLAYER_TWO_COLOR)
    draw_line(GRAPHIC_SIZE, GRAPHIC_SIZE, GRAPHIC_SIZE * (env.board_size - 1), GRAPHIC_SIZE)
    draw_line(GRAPHIC_SIZE, GRAPHIC_SIZE * (env.board_size - 1), GRAPHIC_SIZE * (env.board_size - 1), GRAPHIC_SIZE * (env.board_size - 1))
    set_line_width(1)

    # draw pegs
    for i in range(env.board_size):
        for j in range(env.board_size):

            # get the peg value
            peg = env.board[i, j, 0]

            # set colors
            set_color(PEG_OUTLINE_COLOR)
            set_fill_color(EMPTY_PEG_COLOR)
            if peg == 1:
                set_fill_color(PLAYER_ONE_COLOR)
            elif peg == -1:
                set_fill_color(PLAYER_TWO_COLOR)

            # if one player has won, override the colors
            if env.winner == 1:
                set_fill_color(PLAYER_ONE_COLOR)
            elif env.winner == -1:
                set_fill_color(PLAYER_TWO_COLOR)
            elif env.winner == 0:
                set_fill_color(GUIDE_LINE_COLOR)

            # draw peg
            draw_ellipse(i*GRAPHIC_SIZE + GRAPHIC_SIZE/2, j*GRAPHIC_SIZE + GRAPHIC_SIZE/2, GRAPHIC_SIZE/4, GRAPHIC_SIZE/4)
            
    # draw bridges
    set_line_width(GRAPHIC_SIZE/8) # set line  width
    for i in range(env.board_size): 
        for j in range(env.board_size):
            for k in range(1,5): # only right-facing vector maps required

                # check if there is a bridge starting here
                bridge = env.board[i, j, k] 

                if bridge == 1 or bridge == -1:

                    # determine which and set color
                    if bridge == 1:
                        set_color(PLAYER_ONE_COLOR)
                    elif bridge == -1:
                        set_color(PLAYER_TWO_COLOR)

                    # determine bridge endpoints
                    pos1 = (i, j)
                    pos2 = (0, 0)

                    if k == 1: 
                        pos2 = (i+1, j+2)
                    elif k == 2:
                        pos2 = (i+2, j+1)
                    elif k == 3:
                        pos2 = (i+2, j-1)
                    elif k == 4:
                        pos2 = (i+1, j-2)

                    # draw the bridge
                    draw_line(pos1[0] * GRAPHIC_SIZE + GRAPHIC_SIZE / 2, pos1[1] * GRAPHIC_SIZE + GRAPHIC_SIZE / 2, pos2[0] * GRAPHIC_SIZE + GRAPHIC_SIZE / 2, pos2[1] * GRAPHIC_SIZE  + GRAPHIC_SIZE / 2)

    set_line_width(1) # reset line width to default                      


def draw_heatmap(matrix, min_q_val=-1000000):
    
    set_line_width(0)
    clear_device()

    matrix_with_positions = []

    for i in range(len(matrix)):
        mwp_col = []
        for j in range(len(matrix)):
            mwp_col.append(((i,j), matrix[i][j]))
        matrix_with_positions.append(mwp_col)

    matrix_with_positions = np.array(matrix_with_positions, dtype=[('position', tuple), ('q_val', float)])
    matrix_with_positions = np.sort(matrix_with_positions, axis=None, order='q_val')
    total = len(matrix_with_positions)

    for i in range(total):
            
        position, q_val = matrix_with_positions[i]
        x, y = position

        percentile = i/total

        heat_color = Color.WHITE

        if q_val == min_q_val:
            heat_color = HEAT_COLOR[0]
        elif percentile < 3/4:
            heat_color = HEAT_COLOR[0]
        elif percentile < 7/8:
            heat_color = HEAT_COLOR[1]
        elif percentile < 16/17:
            heat_color = HEAT_COLOR[2]
        elif percentile < 34/35:
            heat_color = HEAT_COLOR[3]
        elif percentile < 69/70:
            heat_color = HEAT_COLOR[4]
        elif percentile < 142/143:
            heat_color = HEAT_COLOR[5]
        elif percentile < 571/572:
            heat_color = HEAT_COLOR[6]
        else:
            heat_color = HEAT_COLOR[7]

        set_color(heat_color)
        set_fill_color(heat_color)

        min_x = x * GRAPHIC_SIZE
        min_y = y * GRAPHIC_SIZE

        draw_rect(min_x, min_y, min_x + GRAPHIC_SIZE, min_y + GRAPHIC_SIZE)

    set_line_width(1)

def mainloop():
    loop = True
    set_line_width(1)

    # begin graphics loop
    while loop:

        renderEnvironment(environment, False)

        # process input
        if environment.winner == None:
            valid_peg = False
            while not valid_peg:
                # wait for a mouse click
                mouse = get_click()

                # get the coordinates of the mouse click
                pegX = int((mouse.x)/(GRAPHIC_SIZE))
                pegY = int((mouse.y)/(GRAPHIC_SIZE))

                # attempt to add peg and, if successful, break the while loop
                click_output = environment.add_peg((pegX, pegY))
                valid_peg = click_output
    

def initialize_graphics(env):
    init_graph(env.board_size * GRAPHIC_SIZE, env.board_size * GRAPHIC_SIZE)
    set_render_mode(RenderMode.RENDER_AUTO)

def main():
    initialize_graphics(environment)
    mainloop()
    # close_graph()


if __name__=="__main__":
    main()