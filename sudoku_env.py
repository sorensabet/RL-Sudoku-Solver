import gym
from gym import spaces
import numpy as np
import sys
from termcolor import colored, cprint
import copy 
import numpy as np
import math
import os 
import helper
import time
import pandas as pd

def cls():
    os.system('cls' if os.name=='nt' else 'clear')

class cell:
    def __init__(self):
        self.value = np.nan
        self.possible_values = [1,2,3,4,5,6,7,8,9]
        self.temporary_possible_values = [1,2,3,4,5,6,7,8,9] # Helps keep the solver interpretable, especially in cases that would be obvious to humans
        self.impossible_values = []
        self.row = np.nan # Row number, ranges from 1-9 as defined below 
        self.col = np.nan # Column number, ranges from 1-9 as defined below
        self.box = np.nan # Box number, ranges from 1-9 as defined below 
        self.cell_id = (row,col)
        self.multi_cell_excluded = False # Has the cell already been excluded based on the multi-cell function analysis? If so, don't show it again 

# Row, col, and box are essentially the same class. I can simplify the code by only using one class for these 3. 
class row: 
    def __init__(self):
        self.position = np.nan  # ranges from 1-9, top to bottom 
        self.cells = []
        self.impossible_values = []

class col: 
    def __init__(self):         # ranges from 1-9, left to right 
        self.position = np.nan 
        self.cells = []
        self.impossible_values = []
        
class box:
    def __init__(self):
        self.position = np.nan # ranges from 1-9, 1 2 3 | 4 5 6| 7 8 9 
        self.cells = []
        self.impossible_values = []
    
# Can reset previous multi_cell_description flag inside solve. 
        
class grid:
    def __init__(self):
        self.complete = False 
        self.correct = False
        self.rows = []
        self.cols = []
        self.boxes = []
        self.unsolved_cells = []
        self.solved_cells = []
        self.changed_during_iteration = False # Infinite loop if no changes to gameboard after iteration 
        self.nmpy_grid = None # Numpy representation of the grid
    
    def print_grid(self):
        top_row = '┌─────────┬─────────┬─────────┐'
        mid_row = '├─────────┼─────────┼─────────┤'
        bot_row = '└─────────┴─────────┴─────────┘'
        
        print(top_row)
        for i in range(0,9):
            row_string = '|'
            for j in range(0,9):
                cell_val = self.rows[i].cells[j].value
                if (np.isnan(cell_val)):
                    cell_val = ' '
                else:
                    cell_val = int(cell_val)
                    
                row_string += ' ' + str(cell_val) + ' ' 
                if (j == 2 or j == 5 or j == 8):
                    row_string += '|'
                #if (j % 3 == 0):
                #    row_string += '|'
            print(row_string)
            if (i == 2 or i == 5):
                print(mid_row)
        print(bot_row)
        
    def explain_temporary_exclusions(self, cell, impossible_values, celltype, cellnum):
        exc_string = ''
        exc_values = []
        exc_count = 0
        pronoun = '' 
        
        for idx, num in enumerate(impossible_values):
            try:
                cell.temporary_possible_values.remove(num)
                exc_values.append(num)
                exc_count += 1 
                #print('Succesfully removed ' + str(num) + ' from possible values!')
            except ValueError: 
                continue
            
        for idx, num in enumerate(exc_values):
            if (exc_count == 1):
                exc_string += str(num)
                pronoun = 'it is'
                break
            elif (exc_count == 2):
                if (idx == 0):
                    exc_string += str(num) + ' or '
                else:
                    exc_string += str(num) 
                pronoun = 'they are'
            else:
                pronoun = 'they are'
                if (idx == len(impossible_values)-1):
                    exc_string += 'or ' + str(num) 

                else:
                    exc_string += str(num) + ', '

        if (exc_count > 0):
            #print('Cannot be ' + exc_string + ' because ' + pronoun + ' already in ' + celltype + ' # ' + str(cellnum))
            cell.temporary_possible_values.sort()
            self.changed_during_iteration = True
        
    def explain_exclusions(self, cell, impossible_values, celltype, cellnum):
        exc_string = ''
        exc_values = []
        exc_count = 0
        pronoun = ''
        
        #print('Now inside explain exclusions!')
        #print('Impossible values inside column: ' + str(impossible_values))
        #print('Possible values inside cell: ' + str(cell.possible_values))
        
        for idx, num in enumerate(impossible_values):
            try:
                cell.possible_values.remove(num)
                exc_values.append(num)
                exc_count += 1 
                #print('Succesfully removed ' + str(num) + ' from possible values!')
            except ValueError: 
                #print('Error!')
                continue
        
        for idx, num in enumerate(exc_values):
            if (exc_count == 1):
                exc_string += str(num)
                pronoun = 'it is'
                break
            elif (exc_count == 2):
                if (idx == 0):
                    exc_string += str(num) + ' or '
                else:
                    exc_string += str(num) 
                pronoun = 'they are'
            else:
                pronoun = 'they are'
                if (idx == len(impossible_values)-1):
                    exc_string += 'or ' + str(num) 

                else:
                    exc_string += str(num) + ', '
       
        #print('exclusion count: ' + str(exc_count))
        
        if (exc_count > 0):
            # print('Cannot be ' + exc_string + ' because ' + pronoun + ' already in ' + celltype + ' # ' + str(cellnum))
            cell.possible_values.sort()
            self.changed_during_iteration = True
        
    def multi_cell_compare(self, cell, curr_obj, obj_name):
            # If two cells in the same row or column or box have n possible values that are the same, then they can be excluded from all other cells in that row and column and box. 
            # Step 1. Loop through all other cells in that row and column and box
            # Step 2. Compare possible values in these cells to possible values in the current cell 
            # Step 3. If the possible values match, then exclude those possible values from all other cells in the row (if searching over row) or column (if searching over column) or box (if searching over box)
            # The above works in a two number case, but what if there are more numbers? 
            # Suppose there were 3 numbers that I could exclude. 
            # Step 1. Count number of cells that have the same possible values 
            # Step 2. Count number of possible values. If this matches the number of cells, 
            # Step 3. Go to all unsolved cells that don't have the same possible values, and remove possible values from them. 
            
#            print('')
#            print('Now inside multi_cell_compare, examining ' + obj_name)
#            print('Current cell: ' + str(cell.cell_id))
            
            count = 0
            cells_to_ignore = [cell]
            cells_to_ignore_ids = [cell.cell_id]
            cells_to_update = []
            cells_to_update_ids = []
            
            for curr_cell in curr_obj.cells: 
                if (np.isnan(curr_cell.value) == False):
                    continue
                if (curr_cell == cell): # If we are at the same cell
                    count += 1
                    continue
                
                # I need a way to update the current cells possible values based on the box, row, or column being considered 
                if (obj_name == 'col'):
                    item = cell.col
                elif (obj_name == 'row'):
                    item = cell.row
                elif (obj_name == 'box'):
                    item = cell.box
                
                self.explain_exclusions(curr_cell, curr_cell.impossible_values, obj_name, item)
                
                # print('Comparison cell: ' + str(curr_cell.cell_id))
                # print('Current cell possible values: ' + str(cell.possible_values))
                # print('Comparison cell possible values: ' + str(curr_cell.possible_values))
                
                if (cell.possible_values == curr_cell.possible_values):
                    cells_to_ignore.append(curr_cell)
                    cells_to_ignore_ids.append(curr_cell.cell_id)
                    count += 1
                else:
                    #if (curr_cell.multi_cell_excluded == False):
                    cells_to_update.append(curr_cell)
                    cells_to_update_ids.append(curr_cell.cell_id)
            # print('count: ' + str(count))
            # print('cells_to_ignore: ' + str(len(cells_to_ignore)))
            # print('cells_to_ignore_ids: ' + str(cells_to_ignore_ids))
            
            explanation_string = 'Since cells ' + str(sorted(cells_to_ignore_ids)) + ' can each only contain one of ' + str(cell.possible_values) + ', then cell(s) ' + str(cells_to_update_ids) + ' cannot contain these values.' 
            
            if (len(cells_to_ignore)==count and count > 1 and len(cells_to_ignore[0].possible_values) == count):
                for curr_cell in cells_to_update:
                    if (np.isnan(curr_cell.value) == False):
                        continue
                    #print('Curr_cell: ' + str(curr_cell.cell_id) + ' possible values: ' + str(curr_cell.possible_values))
                    #print('Curr_cell multi_cell_excluded: ' + str(curr_cell.multi_cell_excluded))
                    for val in cell.possible_values:
                        try:
                            curr_cell.possible_values.remove(val)
                            curr_cell.multi_cell_excluded = True
                            self.changed_during_iteration = True
                        except ValueError:
                            pass
                    #print('Cell ' + str(curr_cell.cell_id) + ' cannot be any of ' + str(cell.possible_values) + ' because of cells: ' + str(cells_to_ignore_ids))
              
                if (len(cells_to_update_ids) > 0):
                    pass
                    # print(explanation_string)
                    # print('')
            return
    
    def object_cell_exclusions(self, cell, obj, obj_name):
        # Rather than having a specific function for rows, columns, and cells, I will have a generic one for all 3. 
        relevant_cells = [] # stores all other empty cells in row, column, or box
        
#        if (obj_name == 'box'):
#            print('Inside object cell exclusions for ' + obj_name)
        
        for obj_cell in obj.cells:
            if(np.isnan(obj_cell.value) == False or obj_cell == cell):
                continue
            #print('Currently analyzing cell: ' + str(obj_cell.cell_id) + ' for impossible values')
            #print(obj_name + ' cell: ' + str(obj_cell.cell_id) + ' temporary possible values: ' + str(obj_cell.temporary_possible_values))
            self.explain_temporary_exclusions(obj_cell, self.rows[obj_cell.row-1].impossible_values, 'row', obj_cell.row)
            self.explain_temporary_exclusions(obj_cell, self.cols[obj_cell.col-1].impossible_values, 'col', obj_cell.col)
            self.explain_temporary_exclusions(obj_cell, self.boxes[obj_cell.box-1].impossible_values, 'box', obj_cell.box)
            #print('Updated ' + obj_name + ' cell temporary possible values: ' + str(obj_cell.temporary_possible_values))
            relevant_cells.append(obj_cell)
            
#        if (len(relevant_cells) > 0):
#            for c in relevant_cells:
#                print('Relevant cell: ' + str(c.cell_id))
        
        for val in cell.possible_values:  # 2 or 8
            #print('Now checking if cell : ' + str(cell.cell_id) + ' is ' + str(val))
            is_value_impossible_in_other_cells = True
            for rel_cell in relevant_cells:
                #print('Relevant cell under consideration: ' + str(rel_cell.cell_id))
                #print('Relevant cell possible values: ' + str(rel_cell.temporary_possible_values))
                if (val in rel_cell.temporary_possible_values):
                    is_value_impossible_in_other_cells = False
                    #print('Since ' + str(val) + ' could also be in cell: ' + str(rel_cell.cell_id) + ', we cannot be certain that it is in cell: ' + str(cell.cell_id))
                    break
            if (is_value_impossible_in_other_cells == False):
                continue
            else:
                # We know the current value of interest cannot be in any of the other empty cells, therefore it must go into the current cell
                # print('Since ' + str(val) + ' could not be in any other cell(s) in this ' + obj_name + ', we know that it must be in cell: ' + str(cell.cell_id))
                cell.possible_values = [val]
                # Reset relevant cell possible temporary values
                for rel_cell in relevant_cells:
                    # print(str(rel_cell.cell_id) + ' possible values: ' + str(rel_cell.temporary_possible_values))
                    rel_cell.temporary_possible_values = [1,2,3,4,5,6,7,8,9] # Reset temporary possible values for now
                return        
    
    def dual_x_wing(self, cell, obj, obj_name):
        #Row case: Only two possible positions for the same value in two rows, and they are each in the same column 
        #Col case: Only two possible positions for the same value in two cols, and they are each in the same row 
        
        for val in cell.possible_values: 
            val_count = 1
            rel_cells = []
            for obj_cell in obj.cells:
                if (val_count >= 3):
                    break
                if (np.isnan(obj_cell.value) == False or cell == obj_cell):
                    continue 
                if (val in obj_cell.possible_values):
                    val_count += 1
                    rel_cells.append(obj_cell)
                    
            if (len(rel_cells) == 1): 
                # If obj_name = rows, I need to select columns 
                # If obj_name = cols, I need to select rows
                obj1 = None
                obj2 = None
                
                if (obj_name == 'row'):
                    obj1 = self.cols[cell.col-1]
                    obj2 = self.cols[rel_cells[0].col-1]
                else:
                    obj1 = self.rows[cell.row-1]
                    obj2 = self.rows[rel_cells[0].row-1]
                
                for i in range(0,9):  # Check the rows/cols in the puzzle
                    if (obj1.cells[i] == cell): # If we are in the row/col we just checked 
                        continue
                    if((np.isnan(obj1.cells[i].value) == False or np.isnan(obj2.cells[i].value) == False)): # If either cell already has a value
                        continue
                    if((val in obj1.cells[i].possible_values == False) or (val in obj2.cells[i].possible_values == False)): # If either cell cannot hold the value of interest 
                        continue 
                    
                    # If we pass the three conditions above, then we know we have a 2nd row where we might be able to identify an x wing 
                    # Next step is to check the cells in the 2nd row, and see if the value of interest can only appear in two places 
                    
                    alt_obj = None 
                    
                    if (obj_name == 'row'):
                        alt_obj = self.rows[i]
                    else:
                        alt_obj = self.cols[i]
                    
                    val_count_2 = 0
                    rel_cells_2 = []
                    
                    for c in alt_obj.cells:
                        if (np.isnan(c.value) == False):
                            continue
                        
                        row_imp = self.rows[c.row-1].impossible_values
                        col_imp = self.cols[c.col-1].impossible_values
                        box_imp = self.boxes[c.box-1].impossible_values
                        
                        if (val in row_imp or val in col_imp or val in box_imp):
                            continue
                        if (val in c.possible_values):
                            val_count_2 += 1
                            rel_cells_2.append(c)
                            
                    if (val_count_2 == 2):
                        
                        # Row case: Check if the columns of the two cells of interest are the same as the columns from the first row 
                        # Col case: Check if the rows fo the two cells of interest are the samee as the rows from the first column
                        
                        x_wing_found = False
                        
                        if (obj_name == 'row'):
                            rc1 = rel_cells_2[0].cell_id[1]
                            rc2 = rel_cells_2[1].cell_id[1]
                            
                            if ((rc1 == cell.col or rc1 == rel_cells[0].col) and (rc2 == cell.col or rc2 == rel_cells[0].col)):
                                # print('Cells ' + str(cell.cell_id) + ', ' + str(rel_cells[0].cell_id) + ', ' + str(rel_cells_2[0].cell_id) + ', and ' + str(rel_cells_2[1].cell_id) + ' form an x-wing, thus ' + str(val) + ' cannot be in any other cells of columns ' + str(cell.col) + ' or ' + str(rel_cells[0].col))
                                x_wing_found = True    
                        else:
                            rc1 = rel_cells_2[0].cell_id[0]
                            rc2 = rel_cells_2[1].cell_id[0]

                            if ((rc1 == cell.row or rc1 == rel_cells[0].row) and (rc2 == cell.row or rc2 == rel_cells[0].row)):
                                # print('Cells ' + str(cell.cell_id) + ', ' + str(rel_cells[0].cell_id) + ', ' + str(rel_cells_2[0].cell_id) + ', and ' + str(rel_cells_2[1].cell_id) + ' form an x-wing, thus ' + str(val) + ' cannot be in any other cells of rows ' + str(cell.row) + ' or ' + str(rel_cells[0].row))
                                x_wing_found = True             
                                
                        if (x_wing_found == True):
                            for c in obj1.cells:
                                if (c == cell or c == rel_cells_2[0] or c==rel_cells[0] or c == rel_cells_2[1]):
                                    continue
                                try:
                                    c.possible_values.remove(val)
                                    # print('Removed ' + str(val) + ' from cell ' + str(c.cell_id))
                                    if(len(c.possible_values) == 1):
                                        self.solve_cell(c)
                                except ValueError:
                                    continue
                                
                            for c in obj2.cells:
                                if (c == cell or c == rel_cells_2[0] or c == rel_cells[0] or c == rel_cells_2[1]):
                                    continue
                                try:
                                    c.possible_values.remove(val)
                                    # print('Removed ' + str(val) + ' from cell ' + str(c.cell_id))
                                    if(len(c.possible_values) == 1):
                                        self.solve_cell(c)
                                except ValueError:
                                    continue
                            break
                                
                        
                            
    def x_wing(self, cell, obj, obj_name):
        for val in cell.possible_values:            
            val_count = 1
            rel_cells = []
            for obj_cell in obj.cells:
                if (np.isnan(obj_cell.value) == False or cell==obj_cell):
                    continue
                if (val in obj_cell.possible_values):
                    val_count += 1
                    rel_cells.append(obj_cell)
                
            if(len(rel_cells) == 1):
                col1 = self.cols[cell.col-1]
                col2 = self.cols[rel_cells[0].col-1]
                  
                for r in range(0,9): # Check the rows in the puzzle
                    if(col1.cells[r] == cell): # If we are the row we just checked
                        continue
                    if((np.isnan(col1.cells[r].value) == False or np.isnan(col2.cells[r].value) ==False)): # If either cell already has a value
                        continue
                    if((val in col1.cells[r].possible_values == False) or (val in col2.cells[r].possible_values == False)): # If either cell cannot hold the value of interest 
                        continue 
        
                      # If we pass the three conditions above, then we know we have a 2nd row where we might be able to identify an x wing 
                      # Next step is to check the cells in the 2nd row, and see if the value of interest can only appear in two places 
                      
                    row2 = self.rows[r]
                    val_count_2 = 0
                    rel_cells_2 = []                      
                                            
                    for rc in row2.cells:
                        if (np.isnan(rc.value) == False):
                            continue
                        
                        row_imp = self.rows[rc.row-1].impossible_values
                        col_imp = self.cols[rc.col-1].impossible_values
                        box_imp = self.boxes[rc.box-1].impossible_values
                        
                        if (val in row_imp or val in col_imp or val in box_imp):
                            continue
                        
                        if(val in rc.possible_values):
                            val_count_2 += 1
                            rel_cells_2.append(rc)
                            # I first need to update rc possible values based on other things in the column. 
                                                        
                    if (val_count_2 == 2): # I need to make sure the columns were the same 
                        # For every cell in column 1 and column 2, except for cell, obj_cell, rel_cells_2[0] and rel_cells_2[1]
                        # I can remove the current val from the list of possible values
                        
                        # Check if the columns of the two cells of interest are the same as the columns from the first row 
                        rc1 = rel_cells_2[0].cell_id[1]
                        rc2 = rel_cells_2[1].cell_id[1]
                        
                        if ((rc1 == cell.col or rc1 == rel_cells[0].col) and (rc2 == cell.col or rc2 == rel_cells[0].col)):
                            #print('Cells ' + str(cell.cell_id) + ', ' + str(rel_cells[0].cell_id) + ', ' + str(rel_cells_2[0].cell_id) + ', and ' + str(rel_cells_2[1].cell_id) + ' form an x-wing, thus ' + str(val) + ' cannot be in any other cells of columns ' + str(cell.col) + ' or ' + str(rel_cells[0].col))
                            # Apply x-wing logic
                            # For all cells in columns a and b that arent the cells of interest, remove value from possible values 
                            for c in col1.cells:
                                if (c == cell or c == rel_cells_2[0] or c == rel_cells[0] or c == rel_cells_2[1]):
                                    continue
                                try:
                                    c.possible_values.remove(val)
                                    #print('Removed ' + str(val) + ' from cell ' + str(c.cell_id))
                                    if(len(c.possible_values) == 1):
                                        self.solve_cell(c)
                                except ValueError:
                                    continue
                            for c in col2.cells:
                                if (c == cell or c == rel_cells_2[0] or c == rel_cells[0] or c == rel_cells_2[1]):
                                    continue
                                try:
                                    c.possible_values.remove(val)
                                    #print('Removed ' + str(val) + ' from cell ' + str(c.cell_id))
                                    if(len(c.possible_values) == 1):
                                        self.solve_cell(c)
                                except ValueError:
                                    continue
                            break                   
                                                     
    
    def solve_cell(self, cell): 
        if (np.isnan(cell.value) == False):
            return 
        
        cls()
        self.print_grid()
        
        # print('Now trying to solve cell: ' + str(cell.cell_id))
        # print('Starting possible values: ' + str(cell.possible_values))
        
            
        # easier to understand code if I give the impossible values names
        row = self.rows[cell.row-1]
        col = self.cols[cell.col-1]
        box = self.boxes[cell.box-1]
        
        nums_in_row = row.impossible_values
        nums_in_col = col.impossible_values
        nums_in_box = box.impossible_values

        if (len(cell.possible_values) != 1):
            self.explain_exclusions(cell, nums_in_row, 'row', cell.row)
        if (len(cell.possible_values) != 1):
            self.explain_exclusions(cell, nums_in_col, 'column', cell.col)
        if (len(cell.possible_values) != 1):
            self.explain_exclusions(cell, nums_in_box, 'box', cell.box)
        # print('Possible values: ' + str(cell.possible_values))
        
        if (len(cell.possible_values) != 1):
            self.object_cell_exclusions(cell, row, 'row')
        if (len(cell.possible_values) != 1):
            self.object_cell_exclusions(cell, col, 'column')
        if (len(cell.possible_values) != 1):
            self.object_cell_exclusions(cell, box, 'box')

        if (len(cell.possible_values) != 1):
            self.multi_cell_compare(cell, row, 'row')
        if (len(cell.possible_values) != 1):
            self.multi_cell_compare(cell, col, 'col')
        if (len(cell.possible_values) != 1):
            self.multi_cell_compare(cell, box, 'box')
            
        # Every stragey before this point were things I learned from playing Sudoku over the years.    
        # Every strategy that comes below this line was learned from: http://www.sudokuwiki.org

        if (len(cell.possible_values) != 1):
            self.dual_x_wing(cell, row, 'row')
        if (len(cell.possible_values) != 1):
            self.dual_x_wing(cell, col, 'col')
#        if (len(cell.possible_values) != 1):
#            self.x_wing(cell, col, 'col')

        # If only 1 possible number, set cell value as that, and update row, col, and box possibilities, and remove from solved cells. 
        # otherwise, continue
        if len(cell.possible_values) == 1:
            cell.value = cell.possible_values[0]
            row.impossible_values.append(cell.value) # Probably better to use a set for impossible values, to avoid duplicate values being entered.
            col.impossible_values.append(cell.value)
            box.impossible_values.append(cell.value)
            self.solved_cells.append(cell)
            
            # print('Solved this cell') # input logic to deal with correct cell later. 
            # input ('Press enter to continue')
            
            # Check the number of impossible values in the column and box (and row only if the row # is before the current cells).  
            # If there's only 1 unsolved value, recursively solve that first. 
            if (len(row.impossible_values) == 8):
                for rc in row.cells:
                    if (np.isnan(rc.value) and rc.cell_id[1] < cell.cell_id[1]): 
                        self.solve_cell(rc)
            if (len(col.impossible_values) == 8):
                for cc in col.cells:
                    if (np.isnan(cc.value)):
                        self.solve_cell(cc)
            if (len(box.impossible_values) == 8):
                for bc in box.cells:
                    if (np.isnan(bc.value)):
                        self.solve_cell(bc)


            # Recursion is fancy, but it is not intuitive for a human, and the objective is to have an interpretable sudoku solver. 
#            for row_cell in row.cells:
#                self.solve_cell(row_cell)
#            for col_cell in col.cells:
#                self.solve_cell(col_cell)
#            for box_cell in box.cells:
#                self.solve_cell(box_cell)
            
        else:
            # print('Could not solve cell, continuing to next empty cell...')
            # input('Press enter to continue 1')
            return 

    def iterate(self):
        self.changed_during_iteration = False
        
        
        
        # Loops through each unsolved cell in the game board, and tries to solve.
        for cell in self.unsolved_cells:
            if (np.isnan(cell.value) == False): # In case it was solved in a previous recursion instance
                self.solved_cells.append(cell)
            else:
                self.solve_cell(cell)
        
        for cell in self.solved_cells:
            try:
                self.unsolved_cells.remove(cell)          
            except ValueError:
                continue
        
        # I want a numpy representation of the grid at this turn 
        self.nmpy_grid = np.empty((9,9))
        for i in range(0,9):
            for j in range(0,9):
                self.nmpy_grid[i,j] = self.rows[i].cells[j].value
        
            # Since the more populated a row, grid, or column is, the more information  
            # it provides on a first run, I should dynamically go through them. But that optimization 
            # can come later. 
            
def initialize_grid(brd):
    # Fill the grid with row, column, and boxes 
    for i in range(1,10):
        brd.rows.append(row())
        brd.cols.append(col())
        brd.boxes.append(box())
    
def process_starting_input(inp):
    the_grid = grid()
    initialize_grid(the_grid) # Now grid has rows, columns, and boxes

    # Input: The raw input 
    # Output: Filters the raw game grid, and fills in rows, columns, and boxes with the appropriate cells. 
    for i in range(1,10):        # Row num
        for j in range(1,10):    # Col num
            curr_cell = cell()
            curr_cell.cell_id = (i,j)
            curr_cell.row = i 
            curr_cell.col = j 
            curr_cell.box = 3*int((i-1)/3) + max(1,math.ceil((j)/3))
            curr_cell.value =inp[i-1,j-1]
            if np.isnan(curr_cell.value) == False: # If the cell has a value 
                curr_cell.possible_values = []
                curr_cell.impossible_values = [] 
                the_grid.rows[i-1].impossible_values.append(int(curr_cell.value))
                the_grid.cols[j-1].impossible_values.append(int(curr_cell.value))
                the_grid.boxes[curr_cell.box-1].impossible_values.append(int(curr_cell.value))
            else:
                the_grid.unsolved_cells.append(curr_cell) # Use this list to track empty cells to allow faster iteration
            the_grid.rows[i-1].cells.append(curr_cell)
            the_grid.cols[j-1].cells.append(curr_cell)
            the_grid.boxes[curr_cell.box-1].cells.append(curr_cell)

            #print('Appended cell %s to box %d!' % (str(curr_cell.cell_id), curr_cell.box))
    return the_grid          

def check_legal_moves_remaining(input_grid):
    """
    Checks if a deterministic solver can find at least 1 legal move remaining on the board
    returns True if at least one legal move exists
    returns False otherwise

    """
    
    # Okay. New approach, since old code isn't working
    # Step 1. Find all empty indices in the input grid
    # Step 2. For each empty index, check if any numbers can go there 
    # Step 3. If no numbers can go anywhere for any legal move, the game is over 
    
    
    # If we find a single legal action, we can break out early
    # Otherwise, we search all empty cells for legal actions 
    # If no legal actions remain, we know there's no point playing further
    
    zers = np.where(input_grid==0)
    for ec in list(zip(zers[0],zers[1])):
        for guess in range(1,10):
            is_legal = check_legal_action(guess, ec[0], ec[1], input_grid, verbose=False)
            if (is_legal is True):
                return True 
    return False


# Step 1. Read in puzzle, and fill in rows, grids, and cells. 
def start_solver(input_grid, soln_grid):
    
    input_grid = np.where(input_grid==0, np.nan, input_grid)
    
    game_grid = process_starting_input(input_grid)
    
    #print(game_grid.print_grid())
    iter_count = 1
    
    could_solve = False
    
    while True:
        game_grid.iterate()
        # cls()
        print('Completed iteration # ' + str(iter_count))
        iter_count += 1
        game_grid.print_grid()
        # Check if it has been solved or not
        if (len(game_grid.unsolved_cells) == 0):
            # print('Congratulations! The program was able to solve this Sudoku puzzle')
            #game_grid.print_grid()
            could_solve = True
            break
        elif (game_grid.changed_during_iteration == False):
            #print('No changes were made during the previous iteration - no unique solution found')
            #game_grid.print_grid()
            break
    
    #print(game_grid.print_grid())
    # print(game_grid.nmpy_grid)
    # print(soln_grid)
    
    return could_solve, helper.check_solution_auto(game_grid.nmpy_grid.astype(int), soln_grid)


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

# action[0] = row
# action[1] = col
# action[2] = val (need to +1)


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
    
    # @return
    #   - a copy of the grid to prevent alteration from the user
    #   - a reward: - negative if action leads to an error
    #               - positive if action is correct or grid is resolved
    def step(self, action):
        
        # Action is sampled based on the function in __init__. 
        self.last_action = action
        oldGrid = np.copy(self.grid)
        
        if (action[0], action[1]) in self.init_states_index:
            # if the agent tries to change one of the starting cells of the puzzle, penalize with huge large number because this is illegal
            if self.verbose:
                print('Tried to overwrite initial states of the puzzle! Reward {:d}'.format(-10))
            return np.copy(self.grid), -10, False, None

        # The user can't replace a value that was already set
        if self.grid[action[0], action[1]] != 0:
            # if number already exists at grid
            if self.verbose:
                print('Tried to overwrite existing value! Reward: %d' % (-10))
            return np.copy(self.grid), -10, False, None

        # We add one to the action because the action space is from 0-8 and we want a value in 1-9
        is_legal = check_legal_action(action[2]+1, action[0], action[1], self.grid)
        if (is_legal):
            self.grid[action[0], action[1]] = action[2]+1
            if self.verbose:
                print('Row: %d, Col:%d, Val: %d' % (action[0] + 1, action[1] + 1, action[2]+1))
                print('Game board state after making move: ')
                self.render()

        # check_legal_moves_remaining returns True if there are legal moves remaining
        is_episode_done = not check_legal_moves_remaining(self.grid)
        num_zeros = np.size(self.grid) - np.count_nonzero(self.grid)
        
        # if self.verbose and num_zeros <= 15:
        #     print('Is episode done flag: %s' % (str(is_episode_done)))
        #     print('Game board states after starting solver:')
        #     start_solver(self.grid, self.sol)
            
        #     input('Press enter to continue!')
            #time.sleep(3)
        # is_legal: Check if the current move is legal given the state of the board
        # is_correct: Checks if the current move matches the solution of the puzzle
        # is_finished: Checks if the puzzle is finished 
        # We also need to check for the case where it fails to finish the puzzle
        # because it made an illegal move somewhere
        
        # Reward functions based on correctness and legality
        # If grid is complete or correct, return positive reward
        
        if is_legal: 
            return np.copy(self.grid), 1/self.num_empty, is_episode_done, None
        else:
            return np.copy(self.grid), -1, False, None
        
        
        # if is_finished: 
        #     print('Finished the puzzle \n')
        #     #print(self.grid)
        #     return np.copy(self.grid), 1e3, True, None
        # if is_correct:
        #     return np.copy(self.grid), 20, False, None
        # if is_legal: # If it is unfinished but legal 
        #     return np.copy(self.grid), 2, False, None
        # else:
        #     # If move is wrong, return to old state, and return negative reward
        #     self.grid = oldGrid
        #     return np.copy(self.grid), -5, False, None
        
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
        self.num_empty = np.size(self.grid) - np.count_nonzero(self.grid)
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
