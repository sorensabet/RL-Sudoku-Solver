import numpy as np
import pandas as pd 

def check_legal_action(val, row, col, array): 
    """
        val: The next value predicted by the algorithm 
        row: The row coordinate to input the val
        col: The col coordinate to input the val 
    """
        
    if (array[row,col] != 0):
        print('A value has already been placed there!')
        return False
    
    if (val in array[row]):
        print('Value exists in row!')
        return False 
    
    if (val in array[col]):
        print('Value exists in column!')
        return False 
    
    # To check subgrid, I first need to find the correct subgrid
    row_cor = 3*int(row/3)
    col_cor = 3*int(col/3)
    
    if (val in array[row_cor:row_cor + 3, col_cor:col_cor + 3]):
        print('Value exists in subgrid!')
        return False 
    
    return True 

def check_solution_auto(array, soln): 
    """
        For cases where we have the existing solution
    """
    return (array == soln).all()
        

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

def print_board(array):
    
        """
            Input: (9,9) numpy array representing current grid
            Returns: None
            
            Prints a nice version of the game board
        """

        top_row = '┌─────────┬─────────┬─────────┐'
        mid_row = '├─────────┼─────────┼─────────┤'
        bot_row = '└─────────┴─────────┴─────────┘'
        
        array = np.where(array == 0, np.nan, array)

        print(top_row)

        for i in range(0,9):
            row_string = '|'
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

puzzles = np.load('puzzles.npy')

# for i in range(0,len(puzzles)):
    # print_board(puzzles[i][0])
    # print_board(puzzles[i][1])
    # print('\n\n\n')
    
    #print('%d, %s' % (i, check_correct(puzzles[i][1])))
    # print_board(puzzles[i][0])
    # print(check_legal_action(7,6,2,puzzles[i][0]))
    # break

# Okay. I've structured it in a reasonable way and this should help us develop quickly. 
# What are other useful functions? 

