import gym
from gym import spaces
import numpy as np
import math
import os 
import helper

from gym.envs.registration import register

DEBUG = False
def setDebug(d):
	global DEBUG
	DEBUG = d

register(
    id='Sudoku-v0',
    entry_point=__name__ + ':SudokuEnv',
    reward_threshold = 40.0,
)

error = 2
resolved = 0
unfinished = 1

# Check a solution is correct by checking the 3 contraints on all digits
#   - digit is unique in row
#   - digit is unique in column
#   - digit is unique in square
#  @return
#   - resolved if the grid is resolved
#   - unfinished if the grid is not yet finished
#   - error if one of the contraints is not respected

class SudokuEnv(gym.Env):
		
	metadata = {'render.modes': ['human']}
	last_action = None
    
    # NOTE: VERY IMPORTANT! WE NEED TO INITIALIZE THE ENVIRONMENT BASE GRID IN MAIN!
	# Make a random grid and store it in self.base
	# self.base seems to be a numpy array
    
	def __init__(self, verbose=False):
		# Get a random solution for an empty grid
		self.grid = []
		self.setDebug = False
		self.verbose = verbose


	def step(self, action):
		# Action is sampled based on the function in __init__. 
		self.last_action = action

		old_row = self.agent_state[0]
		old_col = self.agent_state[1]
		zero_elems = helper.get_zero_elements(self.grid)
		old_idx = zero_elems.index((old_row, old_col))
        
        
        
        # With this approach, the agent will never try to change a starting cell,
        # because its movement is limited to empty cells
		if action > 8: 

			if action == 9: # Go left
				new_idx = (zero_elems[old_idx-1]) if (old_idx > 0) else (zero_elems[-1])
			else: # Go right
				new_idx = (zero_elems[old_idx+1]) if (old_idx < len(zero_elems)-1) else (zero_elems[0])
                
			self.agent_state[0] = new_idx[0]
			self.agent_state[1] = new_idx[1]
            
            # Return reward of -1 for moving, to discourage too much movement
			return self.state(), -1, False, None
		else: 
			is_legal = helper.check_legal_action(action+1, old_row, old_col, self.grid)
			if (is_legal):
				self.grid[old_row, old_col] = action+1
        
            # check_legal_moves_remaining returns True if there are legal moves remaining
			is_episode_done = not helper.check_legal_moves_remaining(self.grid)
			#num_zeros = np.size(self.grid) - np.count_nonzero(self.grid)		
            
			if is_legal: 
				if is_episode_done:
					return self.state(), 1/self.num_empty, is_episode_done, None
				else: 
                    # Also need to move the current position to the next empty cell
                    # Set default behaviour of moving right if cell is solved for now
					new_idx = (zero_elems[old_idx+1]) if (old_idx < len(zero_elems)-1) else (zero_elems[0])
                
					self.agent_state[0] = new_idx[0]
					self.agent_state[1] = new_idx[1]
					return self.state(), 1/self.num_empty, is_episode_done, None

			else:
				return self.state(), -1, False, None

	# Replace self.grid with self.base
	# Creating a new grid at every reste would be expensive
	def reset(self):
		self.last_action = None
		self.grid = np.copy(self.base)
		self.num_empty = np.size(self.grid) - np.count_nonzero(self.grid)
        # Randomly sample a starting point
		start_pos = helper.get_rand_star_pos(self.grid)
		self.agent_state = np.array([start_pos[0],start_pos[1]]) # row, col
		return self.state()
		

	def state(self):
		return np.concatenate((np.copy(self.grid).reshape(-1), np.copy(self.agent_state)))

	def render(self, mode='human', close=False):
		### This basically just prints out the game board, and is supposed to highlight 
		### Which cell is being changed by the action. 
		### I will print out the game board for now and add color highlighting. 
		### We don't want agent to change environment. 
		
		coords =  '    1  2  3   4  5  6   7  8  9'
		top_row = '  ┌─────────┬─────────┬─────────┐'
		mid_row = '  ├─────────┼─────────┼─────────┤'
		bot_row = '  └─────────┴─────────┴─────────┘'
		
		# last_action = self.last_action
		array = np.where(self.grid == 0, np.nan, self.grid)
		num_to_let = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

		print(coords)
		print(top_row)

		for i in range(0,9):
			row_string = '%s |' % num_to_let[i]
			for j in range(0,9):                        
				cell_val = array[i,j]
				if (np.isnan(cell_val)):
					cell_val = ' '
				else:
					cell_val = int(cell_val)

				if i == self.agent_state[0] and j == self.agent_state[1]:
					row_string += '[' + str(cell_val) + ']' 
				else:
					row_string += ' ' + str(cell_val) + ' ' 
				if (j == 2 or j == 5 or j == 8):
					row_string += '|'
			print(row_string)
			if (i == 2 or i == 5):
				print(mid_row)
		print(bot_row)
