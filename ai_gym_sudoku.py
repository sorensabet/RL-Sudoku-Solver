import sys
import gym
import sudoku_env
import numpy as np 
import helper
import soren_solver
import time
import math
import random
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import my_pytorch_agents

np.set_printoptions(precision=2)
DEBUG = True
TEST = False

# Load data
puzzles = np.load('puzzles.npy')
num_let_map = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
curr_puzz = puzzles[0][0]

# Model parameters
num_episodes = 500000
batch_size=64

# Initialize environment
env = gym.make('Sudoku-v0')
test_env = gym.make('Sudoku-v0') # Making a copy to not have to worry about overwriting puzzles during testing
env.base = puzzles[0][0] # Unfinished puzzle
env.sol = puzzles[0][1]  # Solution (used to inform reward function)
env.setDebug = DEBUG

state_size = 81 + 2
action_size = 9 + 2
#agent = SudokuAgent(state_size, action_size)
agent = my_pytorch_agents.Agent(lr=0.0001, input_dims=state_size, n_actions=action_size)

is_done=True
puzz_counter = 0

# Metrics to track per episode
enable_tests = False
test_freq = 10      # Evaluate model performance on every n_th puzzle
train_metrics = []
test_metrics = []

# For easier plotting
scores = []
eps_history = []

for e in range(num_episodes):
    #print('Current episode: %d ' % e)
    
    if is_done:
        env.base = puzzles[puzz_counter][0] # Unfinished puzzle
        env.sol = puzzles[puzz_counter][1]  # Solution (used to inform reward function)
        env.verbose = False
    
        # Current position: Get index of all non-zero positions in environment        
        puzz_counter += 1
        is_done = False
        
    # reset state in the beginning of each game
    state = env.reset()
    
    # print('State shape:  %s' % str(state.shape))
    # print('State new shape:  %s' % str(state.reshape(1,-1).shape))
    
    reward_for_ep = 0
    num_iters = 0
    
    while True:
        # turn this on if you want to render
        # env.render()
        
        # Decide action
        
        action = agent.act(state.reshape(1,-1))
        # if DEBUG:
        #     print('Environment before action:')
        #     env.render()
        #     print('Agent State: [%s,%d]' % (num_let_map[env.agent_state[0]], env.agent_state[1]+1))
        #     print('Action: %s' % str(helper.act_desc(action)))

        # Advance the game to the next frame based on the action.
        next_state, reward, done, _ = env.step(action)
        reward_for_ep += reward       

        #act_row = num_let_map[action[0]]
        #act_col = action[1] + 1 
        #act_val = action[2] + 1
        #print('Last action: [%s,%d] ==> %d, Reward: %d\n\n\n' % (act_row, act_col, act_val, reward))
        
        # memorize the previous state, action, reward, and done
        #agent.memorize(state.reshape(1,-1), action, reward, next_state.reshape(1,-1), done)

        agent.learn(state, action, reward, next_state)

        # make next_state the new current state for the next frame.
        state = next_state
        
        # if DEBUG:
        #     print('\nEnvironment after previous action: ')
        #     env.render()
        #     print('Agent State: [%s,%d]' % (num_let_map[env.agent_state[0]], env.agent_state[1]+1))
        #     print('State shape:  %s' % str(state.shape))
        #     print('NState shape: %s' % str(next_state.shape))
        
        
        # Check if the puzzle was completed
        curr_puzz = state[:81].reshape(9,9)
        
        is_solved = np.array_equal(env.sol, curr_puzz)
        num_zeros = np.size(curr_puzz) - np.count_nonzero(curr_puzz)
        
        #print('Episode: %d, Num_iters: %d, Num elements remaining: %d' % (e, num_iters, num_zeros))
        num_iters += 1
        
        #input('Press enter to continue\n\n ------ END OF TIMESTEP -------\n\n')

        
        if done:
            # print the score and break out of the loop
            if is_solved:
                print('Solved puzzle #: %d' % (puzz_counter-1))
            else:
                print('No legal moves remaining for puzzle # %d' % (puzz_counter-1))
            #env.render()
            
            print("\nFinished episode: {}/{}".format(e, num_episodes))
            is_done = True
            break
    
    #print('Finished working on the puzzle!')
    
    # train the agent with the experience of the episode
    # if num_iters >= 63:
    #     agent.replay(64)
    # else:
    #     agent.replay(num_iters-1)
    
    scores.append(reward)
    eps_history.append(agent.epsilon)
    
    if (e % 100 == 0):
        avg_score = np.mean(scores[-100:])
        print('epsiode %d score %0.1f avg score %0.1f epsilon %0.2f' % (e, reward_for_ep, avg_score, agent.epsilon))
    
    filename = 'sudoku_avg_score.png'
    helper.plot_learning_curve(scores, eps_history, filename)
    
    
    # Metrics tracking 
    # train_metrics.append({'episode': e, 'reward': reward_for_ep, 'num_iters': num_iters,
    #                 'avg_reward': reward_for_ep/(num_iters), 'epsilon': agent.epsilon, 
    #                  'solved': is_solved, 'puzzle_num': puzz_counter-1})
        
    # TESTING CURRENT MODEL: 
    # Choose a puzzle randomly that was not seen up to now 

    # if ((enable_tests is True) and (((e+1) % test_freq)==0)):
    #     test_puzz = np.random.randint(puzz_counter, 1000000) # Test randomly on a puzzle we haven't seen yet
    #     test_env.base = puzzles[test_puzz][0] # Unfinished puzzle
    #     test_env.sol = puzzles[test_puzz][1]  # Solution (used to inform reward function)

    #     state = test_env.reset()
    #     test_reward_for_ep = 0
    #     test_solved = False
        
    #     last_guess = None 
    #     last_guess_repeated_count = 0
    #     got_stuck = False
        
    #     test_num_iters = 0
    #     while True:
    #         action = agent.act(state.reshape(1,-1)) # Use a different action function that acts using model
            
    #         if (last_guess == action):
    #             last_guess_repeated_count += 1
    #         else: 
    #             last_guess = action 
    #             last_guess_repeated_count = 0
            
    #         if (last_guess_repeated_count > 50):
    #             got_stuck = True
    #             break
            
            
    #         next_state, reward, done, _ = test_env.step(action)
    #         test_reward_for_ep += reward
            
    #         state = next_state
    #         is_solved = np.array_equal(test_env.sol, state)
            
    #         test_num_iters += 1
    #         if ((done is True) or (test_num_iters > 1000)):
    #             if ((done is True) and (is_solved is True)):
    #                 print('Model solved puzzle #: %d' % (test_puzz))            
    #             elif ((done is True) and (is_solved is False)):
    #                 print('Model ran out of legal moves on puzzle #: %d' % (test_puzz))
    #             elif (done is False):
    #                 print('Model failed to solve puzzle %d in <5k steps!' % (test_puzz))
                    
    #             break
                
    #     if (got_stuck is False):    
    #         test_metrics.append({'num_train_episodes': e, 'reward': test_reward_for_ep,
    #                           'num_iters': test_num_iters, 'avg_reward': test_reward_for_ep/(test_num_iters+1),
    #                           'solved': test_solved, 'puzzle_num': test_puzz})
        
    # print("finished ep {:d} with average reward {:f}, epsilon={:f}".format(e,reward_for_ep/(num_iters+1), agent.epsilon))
    

    # train the agent with the experience of the episode
    # if num_iters >= 63:
    #     agent.replay(64)
    # else:
    #     agent.replay(num_iters-1)

    # Metrics tracking 
    # train_metrics.append({'episode': e, 'reward': reward_for_ep, 'num_iters': num_iters,
    #                 'avg_reward': reward_for_ep/(num_iters), 'epsilon': agent.epsilon, 
    #                  'solved': is_solved, 'puzzle_num': puzz_counter-1})
        
