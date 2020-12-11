import gym
from gym import spaces
import numpy as np
import math
import os 
import helper
import soren_solver

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
		#print('Initializing SudokuEnv from inside my custom file!')


	def step(self, action):
		# Action is sampled based on the function in __init__. 
		self.last_action = action

		old_row = self.agent_state[0]
		old_col = self.agent_state[1]
		zero_elems = helper.get_zero_elements(self.grid)
		old_idx = zero_elems.index((old_row, old_col))
        
        # Checking if human could solve it 
		is_human_solvable, human_value = soren_solver.check_human_can_solve_cell(old_row, old_col, self.grid)
		# Check if human could solve cell 
        # If human can solve the cell, moving is bad 
        # If human can't solve the cell, moving is good 
		# print('Human solvable: %s, Human_val: %s, True_val: %s' % (str(is_human_solvable),str(human_value),self.sol[old_row, old_col]))
        
        # New reward structure: 
        # 1. Check if a human could solve the cell 
        # 2. Check if the value being played is the correct move 
        
        #  POSITIVE REWARDS 
        #    If play_value & is_right=True & is_human_solvable=False:  +6   (DONE)
        #    If play_value & is_right=True & is_human_solvable=True:   +3   (DONE)
        #    If movement   &                 is_human_solvable=False:  +3   (DONE)
        
        #  NEGATIVE REWARDS
        #    If play_value & is_right=False & is_human_solvable=True:  -5   (DONE)
        #    If play_value & is_right=False & is_human_solvable=False: -3   (DONE)
        #    If movement   &                  is_human_solvable=True:  -5   (DONE)
        
        # Environment prevents agent from changing existing cells
        
        # MOVEMENT CASE 
		if action > 8: 
			if action == 9: # Go left
				new_idx = (zero_elems[old_idx-1]) if (old_idx > 0) else (zero_elems[-1])
			else: # Go right
				new_idx = (zero_elems[old_idx+1]) if (old_idx < len(zero_elems)-1) else (zero_elems[0])
                
			self.agent_state[0] = new_idx[0]
			self.agent_state[1] = new_idx[1]
            
			if is_human_solvable: 
			    return self.state(), -5, False, None
			else:
			    return self.state(), +3, False, None
        
        # PLAYING VALUE CASE
		else:
            # IF MOVE IS CORRECT
			if (self.sol[old_row, old_col] == action+1):    
				self.grid[old_row, old_col] = action+1      # Add move to grid
                
				is_episode_done = helper.check_solution_auto(self.grid, self.sol) # Check if grid is complete
                
				#print(is_episode_done)
				#print(is_episode_done == False)
				#print(is_episode_done == False)
                
                # Update the position by moving right
				if (is_episode_done == False):
					#print('Episode not done, cell solved, moving right!')
					new_idx = (zero_elems[old_idx+1]) if (old_idx < len(zero_elems)-1) else (zero_elems[0])        
					self.agent_state[0] = new_idx[0]
					self.agent_state[1] = new_idx[1]
                
				if is_human_solvable == True:    # Human could solve and it guessed right:
					return self.state(), +3, is_episode_done, None
				else:                            # Human could not solve but it guessed right
					return self.state(), +6, is_episode_done, None
            
            # IF MOVE IS INCORRECT
			else: 
				if is_human_solvable == True: 
					return self.state(), -5, False, None 
				else:                
					return self.state(), -3, False, None

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
