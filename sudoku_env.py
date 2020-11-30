import gym
from gym import spaces
import numpy as np
import sys
from termcolor import colored, cprint

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

def check_legal_action(val, row, col, array): 
    """
        val: The next value predicted by the algorithm 
        row: The row coordinate to input the val
        col: The col coordinate to input the val 
    """
    
    # print('Row: %s' %str(array[row]))
    # print('Col: %s' %str(array[col]))
    # print('Val: %s' %str(val))
    
    if (val in array[row]):
        print('Value exists in row!')
        return False 
    
    if (val in array[:,col]):
        print('Value exists in column!')
        return False 
    
    # To check subgrid, I first need to find the correct subgrid
    row_cor = 3*int(row/3)
    col_cor = 3*int(col/3)
    
    if (val in array[row_cor:row_cor + 3, col_cor:col_cor + 3]):
        print('Value exists in subgrid!')
        return False 
    
    return True 

def check_solution_manual(array):
    """
    Parameters
    ----------
    array : (9,9) np.ndarray reprenting completed Sudoku grid

    Returns True if the completed grid is a valid solution, false otherwise
    -------
    
    """
    
    correct = np.array([1,2,3,4,5,6,7,8,9])
        
    # Check each row: 
    for r in range(0,9):
        if ((np.sort(array[r]) == correct).all() == False):
            return False 
    
    for c in range(0,9):
        if ((np.sort(array[:,c]) == correct).all() == False):
            return False 
        
    corners = [(0,0), (0,3), (0,6), (3,0), (3,3), (3, 6), (6,0), (6,3), (6,6)]
    for g in corners:
        sub_grid = array[g[0]:g[0]+3,g[1]:g[1]+3]
        
        if ((np.sort(sub_grid.reshape(-1)) == correct).all() == False):
            print(np.sort(sub_grid.reshape(-1)))
            return False

    return True 

def checkSolution(grid):
	N = len(grid)

	for i in range(N):
		for j in range(N):
			# If a case is not filled, the sudoku is not finished
			if grid[i][j] == 0:
				return unfinished
			
			n = int(N/3)
			
			iOffset = int(i/n*n)
			jOffset = int(j/n*n)
			

			square = grid[ iOffset:iOffset + n , jOffset:jOffset + n].flatten()
			# Check uniqueness
			uniqueInRow    = countItem(grid[i], grid[i, j])  == 1
			uniqueInCol    = countItem(grid[:,j:j+1].flatten(), grid[i, j]) == 1
			uniqueInSquare = countItem(square, grid[i, j]) == 1

			if not (uniqueInRow and uniqueInCol and uniqueInSquare):
				return error

	return resolved


# Count the number of time the item appears in a vector
def countItem(vector, item):
	count = 0
	for item2 in vector:
		if item2 == item: count += 1
	return count


class SudokuEnv(gym.Env):
		
	metadata = {'render.modes': ['human']}
	last_action = None
    
    # NOTE: VERY IMPORTANT! WE NEED TO INITIALIZE THE ENVIRONMENT BASE GRID IN MAIN!
	# Make a random grid and store it in self.base
	# self.base seems to be a numpy array
    
	def __init__(self):
		# box space is continuous. This don't apply to a sudoku grid, but there is no other choices
		self.observation_space = spaces.Box(low=1, high=9, shape=(9, 9))
				
		# This likely generates an x coordinate, a y coordinate, and a number to put into the grid
		self.action_space = spaces.Tuple((spaces.Discrete(9), spaces.Discrete(9), spaces.Discrete(9)))
		
		# Get a random solution for an empty grid
		self.grid = []
	
    
	# @return
	#   - a copy of the grid to prevent alteration from the user
	#   - a reward: - negative if action leads to an error
	#               - positive if action is correct or grid is resolved
	def step(self, action):
        
        # Action is sampled based on the function in __init__. 
		self.last_action = action
		oldGrid = np.copy(self.grid)
        
		# The user can't replace a value that was already set
		if self.grid[action[0], action[1]] != 0:
            # Setting reward to -10 to more strongly penalize 
            # the algorithm when it tries to overwrite existing values
			print('Tried to overwrite existing value! Reward: %d' % (-10))
			return np.copy(self.grid), -10, False, None


        # legal_reward: 1,   correct_reward:  2,    finished_reward:    10
		is_legal = check_legal_action(action[2]+1, action[0], action[1], self.grid)
		is_correct = self.sol[action[0], action[1]] == (action[2] + 1)
		is_finished = check_solution_manual(self.grid)       

		# We add one to the action because the action space is from 0-8 and we want a value in 1-9
		self.grid[action[0], action[1]] = action[2]+1

    
        # is_legal: Check if the current move is legal given the state of the board
        # is_correct: Checks if the current move matches the solution of the puzzle
        # is_finished: Checks if the puzzle is finished 
        # We also need to check for the case where it fails to finish the puzzle
        # because it made an illegal move somewhere
		
        # Reward functions based on correctness and legality
		# If grid is complete or correct, return positive reward
		if is_finished: 
			return np.copy(self.grid), 10, True, None
		if is_correct:
			return np.copy(self.grid), 2, True, None
		if is_legal: # If it is unfinished but legal 
			return np.copy(self.grid), 1, False, None
		else:
			# If move is wrong, return to old state, and return negative reward
			self.grid = oldGrid
			return np.copy(self.grid), -1, False, None
        
        # Original Reward Functions 
		# stats = checkSolution(self.grid)
		# If grid is complete or correct, return positive reward
		#if stats == resolved: # If it is finished
		#	return np.copy(self.grid), 1, True, None
		#elif stats == unfinished: # If it is unfinished but legal 
		#	return np.copy(self.grid), 1, False, None
		#if stats == error:
		#	# If move is wrong, return to old state, and return negative reward
		#	self.grid = oldGrid
		#	return np.copy(self.grid), -1, False, None


	# Replace self.grid with self.base
	# Creating a new grid at every reste would be expensive
	def reset(self):
		self.last_action = None
		self.grid = np.copy(self.base)
		return np.copy(self.grid)


	def render(self, mode='human', close=False):
		### This basically just prints out the game board, and is supposed to highlight 
		### Which cell is being changed by the action. 
		### I will print out the game board for now and add color highlighting. 
		### We don't want agent to change environment. 
		
		coords =  '    1  2  3   4  5  6   7  8  9'
		top_row = '  ┌─────────┬─────────┬─────────┐'
		mid_row = '  ├─────────┼─────────┼─────────┤'
		bot_row = '  └─────────┴─────────┴─────────┘'
		
		last_action = self.last_action
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

				row_string += ' ' + str(cell_val) + ' ' 
				if (j == 2 or j == 5 or j == 8):
					row_string += '|'
			print(row_string)
			if (i == 2 or i == 5):
				print(mid_row)
		print(bot_row)
