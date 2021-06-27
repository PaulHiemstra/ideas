import numpy as np
import random

class _2048:
    def __init__(self, board_size = 4):
        self.board_size = board_size
        self.board = np.zeros([board_size,board_size])
        self.score = 0
        self.done = False
        
        # Randomly setting 2 empty tiles to a value of 2
        x_coors = np.random.choice(np.arange(board_size - 1), size = 2, replace=False)
        y_coors = np.random.choice(np.arange(board_size - 1), size = 2, replace=False)
        self.board[x_coors, y_coors] = 2
    def step(self, action):
        '''
        Move the tiles on the board. Note that the code can only perform the 'left' move, other moves
        are done by rotating the board to the 'left' position before performing the move. 
        '''
        merged_values = np.zeros([self.board_size, self.board_size])  # We need to track which values where created by merging
        self.reward = 0 # Reset the reward score variable for this action, note that self.swap will update this score
        if action == "left":
            self.board = self.swap(self.board, merged_values, 0, 0)
        elif action == "right":
            self.board = np.rot90(self.swap(np.rot90(self.board, 2), merged_values, 0, 0), 2)
        elif action == 'up':
            self.board = np.rot90(self.swap(np.rot90(self.board, 1), merged_values, 0, 0), 3)
        else:
            self.board = np.rot90(self.swap(np.rot90(self.board, 3), merged_values, 0, 0), 1)
            
        # Randomly add a 2 or 4 at an empty location
        board_zero_idxs = np.where(self.board == 0)   # Find empty spaces on the board
        if board_zero_idxs[0].size == 0:              # Cannot place new value, game-over
            self.done = True
        else:
            random_point = np.random.choice(np.arange(board_zero_idxs[0].size))
            self.board[board_zero_idxs[0][random_point], board_zero_idxs[1][random_point]] = np.random.choice(np.array([2,4]))

        return [self.board.flatten(),  # The values on the playing board. With the normal 4x4 2048 board this means 16 variables
                self.reward,           # The reward we got during this episode
                self.done,             # done: Indicates if the game or episode is done or not. 
                {}]                    # info: Additional info such as performance and latency for debugging purposes  
    def reset(self):
        self.board = np.zeros([board_size,board_size])
        self.score = 0
        self.done = False
    def render(self, mode="human"):
        return np.array2string(self.board)
    def close(self):
        pass
    def swap(self, board, merged_values, target_col_number, count):
        '''
        Recursively swap pairs of columns of tiles and merging values appropriately. Note that we ofcourse only swap when there is an empty tile 
        to the left and a non-empty tile to the right. After swapping and merging the current pair of columns, move on to the next pair by
        recusively calling swap again. 
        
        Important to remember here is to think in (parts of) vectors. For example in the swap part we find the indices where we should swap, and then
        for those indices all at once we actually perform the swap in a bit of numpy magic. I got this trick from https://stackoverflow.com/a/47951813. 
        Note that the swapping is done inplace, without an intermediary assignment. This greatly enhances performance. 
        '''
        rows_to_swap = np.where((board[:,target_col_number] == 0) & (board[:,target_col_number + 1] != 0))[0]
        if rows_to_swap.size > 0:
            target_col = np.repeat(target_col_number, rows_to_swap.size)
            board[[rows_to_swap, rows_to_swap], [target_col, target_col + 1]] = board[[rows_to_swap, rows_to_swap],[target_col + 1, target_col]]
            merged_values[[rows_to_swap, rows_to_swap], [target_col, target_col + 1]] = merged_values[[rows_to_swap, rows_to_swap],[target_col + 1, target_col]]  # Swap merged values, so we keep track of those
        
        rows_to_merge = (                                                                    # Which tiles in the current pair of columns should be merged
            np.where((board[:,target_col_number] == board[:,target_col_number + 1]) &        # We we have the same values next to each other
                      (board[:,target_col_number] != 0) &                                    # We do not want to swap zeroes  
                      (merged_values[:,target_col_number] != 1) &                            # Only values that have not been created by merging
                      (merged_values[:,target_col_number + 1] != 1))[0]                      # can be merged. This prevents recursive merging  
        )                     
        if (rows_to_merge.size > 0):
            self.reward += (2 * board[tuple([rows_to_merge, target_col_number])]).sum()  # Reward is the score that the action has accumulated
            self.score += reward  # Score is the total score
            board[tuple([rows_to_merge, target_col_number])] = 2 * board[tuple([rows_to_merge, target_col_number])]
            board[tuple([rows_to_merge, target_col_number + 1])] = 0
            merged_values[tuple([rows_to_merge, target_col_number])] = 1
            
        if count == (self.board_size - 1) * (self.board_size - 1):                 # We've gone through the swapping so many times even the most right number has been pused all the way to the left
            return board
        elif target_col_number == self.board_size - 2:   # At the end of the board, back to start
            return self.swap(board, merged_values, 0, count + 1)
        else:                          # Next column
            return self.swap(board, merged_values, target_col_number + 1, count + 1)