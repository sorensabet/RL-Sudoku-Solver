import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

def plot_learning_curve(scores, epsilons, filename):
    
    x = [i + 1 for i in range(len(scores))]

    
    fig = plt.figure()
    ax = fig.add_subplot(111, label='1')
    ax2 = fig.add_subplot(111, label='2', frame_on=False)
    
    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel('Episodes', color="C0")
    ax.set_ylabel('Epsilon', color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")
    
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0,t-100):(t+1)])
        
    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    
    # Also plot epsilon and avg reward over episodes individually
    plt.figure()
    plt.plot(x, epsilons, 'o')
    plt.grid(True)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon decay over episodes')
    plt.savefig(filename + 'EpsilonOnly.png', bbox_inches='tight')
    plt.xlim([0,1])
    plt.close()
    
    # Avg reward by itself
    plt.figure()
    plt.plot(x, running_avg, 'o')
    plt.grid(True)
    plt.xlabel('Episode')
    plt.ylabel('Avg reward')
    plt.title('Avg reward over episodes')
    plt.savefig(filename+'AvgRewardOnly.png', bbox_inches='tight')
    plt.close()
    

def arr_index_to_action(index):
    """
    Maps an index 0-728 to an action: eg, put 3 in row 5, col 2
    The index will be the output of the NN's predict function because it takes an
    argmax of 729 layers
    """
    
    num_to_fill = index//81 # dont need to +1 here because it is done in the sudoku env source code
    
    flattened_index = index - index//81 * 81
    row = flattened_index//9
    col = flattened_index - row * 9
    
    return [row, col, num_to_fill]

def get_init_states_index(grid):
    """
    Given a starting sudoku board (2d numpy array), get the indices of the grids that are non-zero.
    You want to heavily penalize these grids because the agent is not supposed to change it at all
    """
    m = np.nonzero(grid)
    return list(zip(m[0],m[1]))

def get_zero_elements(grid):
    zero_elems = np.where(grid==0)
    return list(zip(zero_elems[0], zero_elems[1]))

def act_desc(action):
    act_dict = {0: '1', 1: '2', 2: '3', 3: '4',  4: '5',
            5: '6', 6: '7', 7: '8', 8: '9', 9: 'left',
            10: 'right'}
    
    return act_dict[action]

def get_rand_star_pos(grid):
    zero_elems_list = get_zero_elements(grid)
    return zero_elems_list[np.random.randint(0,len(zero_elems_list))]

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
            #print('Checking for legal moves remaining!')
            is_legal = check_legal_action(guess, ec[0], ec[1], input_grid)
            if (is_legal is True):
                return True 
    return False

def check_legal_action(val, row, col, array): 
    """
        val: The next value predicted by the algorithm 
        row: The row coordinate to input the val
        col: The col coordinate to input the val 
    """
    
    # print('\nPrinting from inside check_legal_action')
    # print('val: %s' % str(val))
    # print('row: %s' % str(row))
    # print('col: %s' % str(col))
        
    if (array[row,col] != 0):
        #print('A value has already been placed there!')
        return False
    
    if (val in array[row]):
        #print('Value exists in row!')
        return False 
    
    if (val in array[:,col]):
        #print('Value exists in column!')
        return False 
    
    # To check subgrid, I first need to find the correct subgrid
    row_cor = 3*int(row/3)
    col_cor = 3*int(col/3)
    
    if (val in array[row_cor:row_cor + 3, col_cor:col_cor + 3]):
        #print('Value exists in subgrid!')
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
        
def plot_metrics(tr_s, tr_f, ts, ep, testing=False):
    # Average reward per episode for solved vs unsolved training data
    plt.figure()
    plt.plot(tr_s['epsilon'], tr_s['avg_reward'], 'o', label='solved')
    plt.plot(tr_f['epsilon'], tr_f['avg_reward'], 'o', label='unsolved')
    plt.xlabel('Epsilon (Exploration Rate)')
    plt.ylabel('Avg Reward')
    plt.title('Epsilon vs. avg reward over ' + str(ep) + ' episodes')
    plt.grid(True)
    plt.legend()
    
    # Total reward per episode for solved vs unsolved training data
    plt.figure()
    plt.plot(tr_s['epsilon'], tr_s['reward'], 'o', label='solved')
    plt.plot(tr_f['epsilon'], tr_f['reward'], 'o', label='unsolved')
    plt.xlabel('Epsilon (Exploration Rate)')
    plt.ylabel('Reward')
    plt.title('Epsilon vs. Total reward over ' + str(ep) + ' episodes')
    plt.grid(True)
    plt.legend()
    
    plt.figure()
    plt.plot(tr_s['epsilon'], tr_s['num_iters'], 'o', label='solved')
    plt.plot(tr_f['epsilon'], tr_f['num_iters'], 'o', label='unsolved')
    plt.xlabel('Epsilon (Exploration Rate)')
    plt.ylabel('# of Iterations')
    plt.title('Epsilon vs. Number of iterations/episode in ' + str(ep) + ' episodes')
    plt.grid(True)
    plt.legend()
    
    if ((testing is True) and (len(ts) > 1)):
        plt.figure()
        plt.plot(ts.index + 1, ts['avg_reward'], 'o', label='average reward')
        plt.xlabel('Test #')
        plt.ylabel('Average rewards')
        plt.title('Test # vs. Average Rewards for ' + str(len(ts)) + ' test runs')
        plt.grid(True)
        plt.legend()
    
    plt.show()

#puzzles = np.load('puzzles.npy')

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

