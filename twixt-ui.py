"""
    Implements a User Interface for the Twixt board game
"""

__author__ = "Alexander Knepprath"

import twixt
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

# init board
env = twixt.TwixtEnvironment(24)

def mainloop():
    loop = True
    set_line_width(1)

    # begin graphics loop
    while loop:

        # set background color based on player turn
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
                peg = env.board[0, i,j]

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
        for i in range(1,5): # only right-facing vector maps required
            for j in range(env.board_size):
                for k in range(env.board_size):

                    # check if there is a bridge starting here
                    bridge = env.board[i, j, k] 

                    if bridge == 1 or bridge == -1:

                        # determine which and set color
                        if bridge == 1:
                            set_color(PLAYER_ONE_COLOR)
                        elif bridge == -1:
                            set_color(PLAYER_TWO_COLOR)

                        # determine bridge endpoints
                        pos1 = (j, k)
                        pos2 = (0, 0)

                        if i == 1: 
                            pos2 = (j+1, k+2)
                        elif i == 2:
                            pos2 = (j+2, k+1)
                        elif i == 3:
                            pos2 = (j+2, k-1)
                        elif i == 4:
                            pos2 = (j+1, k-2)

                        # draw the bridge
                        draw_line(pos1[0] * GRAPHIC_SIZE + GRAPHIC_SIZE / 2, pos1[1] * GRAPHIC_SIZE + GRAPHIC_SIZE / 2, pos2[0] * GRAPHIC_SIZE + GRAPHIC_SIZE / 2, pos2[1] * GRAPHIC_SIZE  + GRAPHIC_SIZE / 2)

        set_line_width(1) # reset line width to default                      


        # process input
        if env.winner == None:
            valid_peg = False
            while not valid_peg:
                # wait for a mouse click
                mouse = get_click()

                # get the coordinates of the mouse click
                pegX = int((mouse.x)/(GRAPHIC_SIZE))
                pegY = int((mouse.y)/(GRAPHIC_SIZE))

                # attempt to add peg and, if successful, break the while loop
                click_output = env.add_peg((pegX, pegY))
                valid_peg = click_output
    


def main():
    
    init_graph(env.board_size * GRAPHIC_SIZE, env.board_size * GRAPHIC_SIZE)
    set_render_mode(RenderMode.RENDER_AUTO)
    mainloop()
    # close_graph()


if __name__=="__main__":
    main()