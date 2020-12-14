import gym
from gym import spaces
import numpy as np
import sys
from termcolor import colored, cprint

from gym.envs.registration import register

register(
    id='Sudoku-v0',
    entry_point=__name__ + ':SudokuEnv',
)

error = 2
resolved = 0
unfinished = 1

def check_legal_action(val, row, col, array, verbose=False): 
    """
        val: The next value predicted by the algorithm 
        row: The row coordinate to input the val
        col: The col coordinate to input the val 
    """
    
    # print('Row: %s' %str(array[row]))
    # print('Col: %s' %str(array[col]))
    # print('Val: %s' %str(val))
    
    if (val in array[row]):
        if verbose:
            print('Value exists in row!')
        return False 
    
    if (val in array[:,col]):
        if verbose:
            print('Value exists in column!')
        return False 
    
    # To check subgrid, I first need to find the correct subgrid
    row_cor = 3*int(row/3)
    col_cor = 3*int(col/3)
    
    if (val in array[row_cor:row_cor + 3, col_cor:col_cor + 3]):
        if verbose:
            print('Value exists in subgrid!')
        return False 
    
    return True 

def check_solution_manual(array, verbose=False):
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

def row_col_grid_filled(grid, row, col, num):
    """
    Returns if a rol, col or subgrid is NEWLY filled.
    """
    correct = np.array([1,2,3,4,5,6,7,8,9])
    
    row_filled = False
    col_filled = False
    subgrid_filled = False
    
    grid_temp = np.copy(grid) # must copy or else changes to grid will change the original variable
    grid_temp[row,col] = num
    
    if (np.sort(grid_temp[row]) == correct).all() == True:
        row_filled = True
        
    if (np.sort(grid_temp[:,col]) == correct).all() == True:
        col_filled = True
        
    corner = (row//3*3, col//3*3)
    sub_grid = grid_temp[corner[0]:corner[0]+3,corner[1]:corner[1]+3]
    
    if ((np.sort(sub_grid.reshape(-1)) == correct).all() == True):
        subgrid_filled = True
        
    return np.array([row_filled, col_filled, subgrid_filled])

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
    
    def __init__(self, verbose=False):
        # box space is continuous. This don't apply to a sudoku grid, but there is no other choices
        self.observation_space = spaces.Box(low=1, high=9, shape=(9, 9))
                
        # This likely generates an x coordinate, a y coordinate, and a number to put into the grid
        self.action_space = spaces.Tuple((spaces.Discrete(9), spaces.Discrete(9), spaces.Discrete(9)))
        
        # Get a random solution for an empty grid
        self.grid = []

        # verbosity
        self.verbose = verbose
    
        # initialize base and sol and init_states_index. these variables are manually passed in by the user
        self.base = None
        self.sol = None
        self.init_states_index = None

    def step(self, action):
        
        self.last_action = action
        oldGrid = np.copy(self.grid)
        
        if (action[0], action[1]) in self.init_states_index:
            # if the agent tries to change one of the starting cells of the puzzle, penalize with huge large number because this is illegal
            if self.verbose:
                print('Tried to overwrite initial states of the puzzle! Reward {:d}'.format(-1))
            return np.copy(self.grid), -1, False, None


        filled = row_col_grid_filled(self.grid, action[0], action[1], action[2]) # check if it fills a new row, col, subgrid
        filled_reward = filled.sum() # +1 reward for filling a row, col or subgrid

        is_correct = (self.sol[action[0], action[1]] == action[2]) * 2 # if it is true it will be 2, else 0
        is_legal = check_legal_action(action[2], action[0], action[1], self.grid)
        
        prev_num = self.grid[action[0], action[1]]

        self.grid[action[0], action[1]] = action[2] # take the action. if it is illegal, the if statement at the bottom will return the old state
        is_finished = np.array_equal(self.sol, self.grid)

        if is_finished: 
            print('Finished the puzzle')
            return np.copy(self.grid), 3, True, None

        if is_legal:

            if prev_num != 0:
                # if overriding an existing number
                if prev_num == action[2]:
                    # if the number it tried to override with is the same number as before
                    if self.verbose:
                        print('Tried to override existing value with the same number. Reward: %d' %(-1))
                    return np.copy(self.grid),-1, False, None

                # if it is overriding with a new number
                
                if self.verbose:
                    print('Tried to overwrite existing value.  Reward: %d' %(filled_reward + is_correct))
                return np.copy(self.grid), filled_reward + is_correct, False, None
                
        # if is_correct and is_legal:
        #     return np.copy(self.grid), filled_reward + is_correct, False, None
        if is_correct:
            return np.copy(self.grid), filled_reward + is_correct, False, None
        if is_legal:
            return np.copy(self.grid), 1 + filled_reward , False, None
        else:
            # If move is wrong, return to old state, and return negative reward
            if self.verbose:
                print("Tried to make an illegal move")
                print(action[0], action[1], action[2])
            self.grid = oldGrid
            return np.copy(self.grid), -1, False, None


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
