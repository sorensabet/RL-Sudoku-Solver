# import numpy as np
# import pandas as pd 


import gym
import gym_sudoku
import numpy as np 
import helper
import soren_solver
import time

num_let_map = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

env = gym.make('Sudoku-v0')
puzzles = np.load('puzzles.npy')
curr_puzz = puzzles[0][0]

# This is how we pass in the puzzle to be solved
env.base = puzzles[0][0] # Unfinished puzzle
env.sol = puzzles[0][1]  # Solution (used to inform reward function)

# Needed to initialize
env.reset()

#env.render()


for i in range(1000):
    env.render()

    step_res = env.step(env.action_space.sample()) # take a random action
    # observ = step_res[0] # An environment specific object representing our observation (e.g. Pixel data from Camera, Joint Angles, Joint velocities, Board state) 
    # reward = step_res[1] # Amount of reward achieved by previous action 
    # finish = step_res[2] # Whether it is time to reset the environment
    # info   = step_res[3] # Diagnostic info for debugging
    
    act_row = num_let_map[env.last_action[0]]
    act_col = env.last_action[1] + 1 # We add 1 here because grid columns are visualized from 1-9
    act_val = env.last_action[2] + 1 # We add 1 here because the action space is from 0-8 and we want a value in 1-9


    # Reward of 1 indicates legal move, reward of 2 indicates correct move
    if (step_res[1] == 1 or step_res[1] == 2):
        print('Time Step: %d, Reward: %d' % (i, step_res[1]))
        print('Last action: [%s,%d] ==> %d \n\n\n\n\n' % (act_row, act_col, act_val))
        input('Press enter to continue!')    
    else: 
        print('Time Step: %d, Reward: %d' % (i, step_res[1]))
    print('\n\n\n')
    #input('Press enter to continue!')
    
env.close()
