import gym
from gym import spaces
import numpy as np
import sys
import math

from gym.envs.registration import register

DEBUG = True

register(
    id='Sudoku-v0',
    entry_point=__name__ + ':SudokuEnv',
    reward_threshold=40.0,
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
        if DEBUG: print('Value exists in row!')
        return False

    if (val in array[:, col]):
        if DEBUG: print('Value exists in column!')
        return False

    # To check subgrid, I first need to find the correct subgrid
    row_cor = 3*int(row/3)
    col_cor = 3*int(col/3)

    if (val in array[row_cor:row_cor + 3, col_cor:col_cor + 3]):
        if DEBUG: print('Value exists in subgrid!')
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

    correct = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    # Check each row:
    for r in range(0, 9):
        if ((np.sort(array[r]) == correct).all() == False):
            return False

    for c in range(0, 9):
        if ((np.sort(array[:, c]) == correct).all() == False):
            return False

    corners = [(0, 0), (0, 3), (0, 6), (3, 0), (3, 3),
                (3, 6), (6, 0), (6, 3), (6, 6)]
    for g in corners:
        sub_grid = array[g[0]:g[0]+3, g[1]:g[1]+3]

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

            square = grid[iOffset:iOffset + n, jOffset:jOffset + n].flatten()
            # Check uniqueness
            uniqueInRow = countItem(grid[i], grid[i, j]) == 1
            uniqueInCol = countItem(grid[:, j:j+1].flatten(), grid[i, j]) == 1
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
        self.action_space = spaces.Tuple(
            (spaces.Discrete(9), spaces.Discrete(9), spaces.Discrete(9)))

        # Get a random solution for an empty grid
        self.grid = []

    # @return
    #   - a copy of the grid to prevent alteration from the user
    #   - a reward: - negative if action leads to an error
    #               - positive if action is correct or grid is resolved

    def step(self, action):
        # Action is sampled based on the function in __init__.
        self.last_action = action

        # action[0-13]
        r_impossible = -5
        r_illegal_movement = -5
        r_correct = 5
        r_finish = 10
        r_good_effort = 8
        r_dead = -10

        # r_movement should range from 2 -> -2 # https://www.desmos.com/calculator/kn9tpwdan5
        a = 5; b = -0.5; k = 4
        sigmoid = k / (1 + np.exp(a + b * self.time_since))
        if DEBUG: print(self.time_since, sigmoid)
        r_movement = 2 - sigmoid

        # decrement time value every step
        if action >= 9:
            self.time_since += 1

        if action == 9:  # up
            is_legal = self.position >= 9
            if is_legal:
                if DEBUG: print('action: %s moving up' % (action))
                self.position -= 9
                return self.state(), r_movement, False, None
            else:
                if DEBUG: print('action: %s illegal movement' % (action))
                return self.state(), r_illegal_movement, False, None
        if action == 10:  # down
            is_legal = self.position <= 71
            if is_legal:
                if DEBUG: print('action: %s moving down' % (action))
                self.position += 9
                return self.state(), r_movement, False, None
            else:
                if DEBUG: print('action: %s illegal movement' % (action))
                return self.state(), r_illegal_movement, False, None
        if action == 11:  # left
            is_legal = self.position != 0 and self.position % 9 != 0
            if is_legal:
                if DEBUG: print('action: %s moving left' % (action))
                self.position -= 1
                return self.state(), r_movement, False, None
            else:
                if DEBUG: print('action: %s illegal movement' % (action))
                return self.state(), r_illegal_movement, False, None
        if action == 12:  # right
            is_legal = self.position != 80 and (self.position + 1) % 9 != 0
            if is_legal:
                if DEBUG: print('action: %s moving right' % (action))
                self.position += 1
                return self.state(), r_movement, False, None
            else:
                if DEBUG: print('action: %s illegal movement' % (action))
                return self.state(), r_illegal_movement, False, None

        row = math.floor(self.position / 9)
        col = self.position % 9
        if self.grid[row, col] != 0:
            # print('Tried to overwrite existing value! Reward: %d' % (-10))
            if DEBUG: print('action: %s overwrite' % (action))
            return self.state(), r_impossible, False, None

        is_legal = check_legal_action(action+1, row, col, self.grid)
        if is_legal:
            self.time_since = 0
            self.grid[row, col] = action+1

            if np.all(self.grid != 0):
                if DEBUG: print('action: %s finished game' % (action))
                return self.state(), r_finish, True, None
            else:
                filled = (self.grid != 0).sum()
                r_place = filled/81 * r_good_effort
                if DEBUG: print('action: %s placed a number' % (action))
                return self.state(), r_place, False, None
        else:
            if DEBUG: print('action: %s illegal number' % (action))
            return self.state(), r_dead, True, None

        # legal_reward: 1,   correct_reward:  2,    finished_reward:    10
        # is_correct = self.sol[row, col] == (action + 1)
        # if is_correct:
        # 	# We add one to the action because the action space is from 0-8 and we want a value in 1-9
        # 	self.grid[row, col] = action+1

        # 	is_finished = check_solution_manual(self.grid)
        # 	if is_finished:
        # 		if DEBUG: print('action: %s finish' % (action))
        # 		return self.state(), r_finish, True, None
        # 	else:
        # 		if DEBUG: print('action: %s correct' % (action))
        # 		self.time_since = 0 # reset after correct move
        # 		return self.state(), r_correct, False, None
        # is_legal = check_legal_action(action+1, row, col, self.grid)
        # if is_legal:
        # 	# print('Tried to write the wrong value! Reward: %d' % (0))
        # 	if DEBUG: print('action: %s wrong value' % (action))
        # 	return self.state(), r_good_effort, False, None
        # else:
        # 	if DEBUG: print('action: %s illegal' % (action))
        # 	return self.state(), r_illegal_movement, False, None

        # We add one to the action because the action space is from 0-8 and we want a value in 1-9
        # self.grid[self.agent_state[0], self.agent_state[1]] = action+1
    
        # is_legal: Check if the current move is legal given the state of the board
        # is_correct: Checks if the current move matches the solution of the puzzle
        # is_finished: Checks if the puzzle is finished 
        # We also need to check for the case where it fails to finish the puzzle
        # because it made an illegal move somewhere
        
        # Reward functions based on correctness and legality
        # If grid is complete or correct, return positive reward
        # if is_finished: 
        # 	return np.copy(self.grid), 10, True, None
        # if is_correct:
        # 	return np.copy(self.grid), 2, False, None
        # if is_legal: # If it is unfinished but legal 
        # 	return np.copy(self.grid), 1, False, None
        # else:
        # 	# If move is wrong, return to old state, and return negative reward
        # 	self.grid = oldGrid
        # 	return np.copy(self.grid), -1, False, None
        
        # Original Reward Functions 
        # stats = checkSolution(self.grid)
        # If grid is complete or correct, return positive reward
        # if stats == resolved: # If it is finished
        #	return np.copy(self.grid), 1, True, None
        # elif stats == unfinished: # If it is unfinished but legal 
        #	return np.copy(self.grid), 1, False, None
        # if stats == error:
        #	# If move is wrong, return to old state, and return negative reward
        #	self.grid = oldGrid
        #	return np.copy(self.grid), -1, False, None


    # Replace self.grid with self.base
    # Creating a new grid at every reste would be expensive
    def reset(self):
        self.last_action = None
        self.grid = np.copy(self.base)
        # self.agent_state = np.array([4,4,0]) # row, col, steps since last number
        self.position = 40
        self.time_since = 0
        return self.state()
        

    def state(self):
        pos = np.zeros(82)
        pos[self.position] = 1
        pos[81] = self.time_since
        return np.concatenate((np.copy(self.grid).reshape(-1), pos))

    def render(self, mode='human', close=False):
        # This basically just prints out the game board, and is supposed to highlight 
        # Which cell is being changed by the action. 
        # I will print out the game board for now and add color highlighting. 
        # We don't want agent to change environment. 
        
        coords =  '    1  2  3   4  5  6   7  8  9'
        top_row = '  ┌─────────┬─────────┬─────────┐'
        mid_row = '  ├─────────┼─────────┼─────────┤'
        bot_row = '  └─────────┴─────────┴─────────┘'
        
        # last_action = self.last_action
        array = np.where(self.grid == 0, np.nan, self.grid)
        num_to_let = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

        print(coords)
        print(top_row)

        # row = math.floor(self.position / 9)
        # col = self.position % 9

        for i in range(0,9):
            row_string = '%s |' % num_to_let[i]
            for j in range(0,9):                        
                cell_val = array[i,j]
                if (np.isnan(cell_val)):
                    cell_val = ' '
                else:
                    cell_val = int(cell_val)

                # if i == row and j == col:
                #     row_string += '[' + str(cell_val) + ']' 
                # else:
                row_string += ' ' + str(cell_val) + ' ' 
                if (j == 2 or j == 5 or j == 8):
                    row_string += '|'
            print(row_string)
            if (i == 2 or i == 5):
                print(mid_row)
        print(bot_row)
