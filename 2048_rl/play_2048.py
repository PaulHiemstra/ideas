import curses
import os
import sys
from datetime import datetime
import numpy as np
from _2048 import _2048

def print_screen(game, win, start=None):
    win.addstr(np.array2string(game.board))
    win.addstr('\n\nscore: ' + str(game.score))
    if not start == None:
        win.addstr('\ntime: ' + str(datetime.now() - start))
        
def main(win):
    game = _2048(int(sys.argv[1]))    
    
    win.nodelay(True)
    key=""
    win.clear()                
    print_screen(game, win)
    gameover = False
    while not gameover: 
        start = datetime.now()
        try:                 
           key = win.getkey()         
           win.clear()                
           if key == 'KEY_LEFT':
              gameover = game.make_move('left')
           elif key == 'KEY_RIGHT':
              gameover = game.make_move('right')
           elif key == 'KEY_UP':
              gameover = game.make_move('up')
           elif key == 'KEY_DOWN':
              gameover = game.make_move('down')
           print_screen(game, win, start)
        except curses.error:
           # No input   
           pass         

curses.wrapper(main)