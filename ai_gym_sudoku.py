import sys
import gym
import sudoku_env
import numpy as np 
import helper
import soren_solver
import time
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
from my_pytorch_agents import DDQNAgent

np.set_printoptions(precision=2)
DEBUG = True
TEST = False

# Load data
puzzles = np.load('puzzles.npy')
num_let_map = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
curr_puzz = puzzles[0][0]

# Model parameters
batch_size=64
best_score = -np.inf
num_episodes = 800000
load_checkpoint = False 
chk_freq = 100             # Evaluate model performance and generate metrics after every 100th episode

state_size = 81 + 2
action_size = 9 + 2

prms = {'algo': 'DDQNAgent', 'gamma': 0.99, 'epsilon': 1.0,
          'lr': 0.0001, 'input_dims': state_size, 'n_actions': action_size,
          'mem_size': 50000, 'eps_min': 0.25, 'batch_size': batch_size, 
          'replace': 1000, 'eps_dec': 1e-7, 'bst_chkpt_dir': 'models/bst_chkpts/', 
          'reg_chkpt_dir': 'models/reg_chkpts/',
          'env_name': 'Sudoku_11acts_6rewards', 'right_guess': 0.5}

agent = DDQNAgent(gamma=prms['gamma'], 
                  epsilon=prms['epsilon'], 
                  lr=prms['lr'], 
                  input_dims=prms['input_dims'], 
                  n_actions=prms['n_actions'], 
                  mem_size= prms['mem_size'], 
                  eps_min=prms['eps_min'], 
                  batch_size=prms['batch_size'], 
                  replace=prms['replace'], 
                  eps_dec=prms['eps_dec'], 
                  bst_chkpt_dir=prms['bst_chkpt_dir'], 
                  reg_chkpt_dir=prms['reg_chkpt_dir'],
                  algo=prms['algo'], 
                  env_name=prms['env_name'], 
                  right_guess=prms['right_guess'])

with open('models/exp_design.txt', 'a') as file:
    file.write(str(prms) + '\n')

if load_checkpoint: 
    agent.load_models()
    
filename = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' +\
    '_' + str(num_episodes) + 'games'
figure_file = 'plots/' + filename + '.png'
data_file = 'data/' + filename + '.png'
    
# Initialize environment
env = gym.make('Sudoku-v0')
env.base = puzzles[0][0] # Unfinished puzzle
env.sol = puzzles[0][1]  # Solution (used to inform reward function)
env.setDebug = DEBUG
puzz_counter = 0

test_env = gym.make('Sudoku-v0')

# Metrics to track per episode
enable_tests = True
train_metrics, test_metrics = [], []
scores, eps_history, steps_array = [], [], []


is_done=True

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
      
        action = agent.act(state.reshape(1,-1))
        
        # if DEBUG:
        #     print('Environment before action:')
        #     env.render()
        #     if (action == 9): # Move left 
        #         print('ACTION: MOVE_LEFT  from [%s,%d]' % (num_let_map[env.agent_state[0]], env.agent_state[1]+1))
        #     elif (action==10):
        #         print('ACTION: MOVE_RIGHT from [%s,%d]' % (num_let_map[env.agent_state[0]], env.agent_state[1]+1))
        #     else:
        #         print('ACTION: [%s, %d] ==> %d' % (num_let_map[env.agent_state[0]], env.agent_state[1]+1, action+1)) 
            
        # Advance the game to the next frame based on the action.
        next_state, reward, done, _ = env.step(action)
        reward_for_ep += reward
        
        if not load_checkpoint: 
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()

        state = next_state
        
        # if DEBUG:
        #     print('Reward: %0.2f\n\n' % reward)
        #     # else:
        #     #     print('Last action: Move to [%s, %d]' % (num_let_map[env.agent_state[0]], env.agent_state[1]+1))
                
        #     #print('Environment after previous action: ')
        #     #env.render()
        #     print('Agent Position: [%s,%d]' % (num_let_map[env.agent_state[0]], env.agent_state[1]+1))
        #     #input('Batman')
        
        # Check if the puzzle was completed
        curr_puzz = state[:81].reshape(9,9)
        
        is_solved = np.array_equal(env.sol, curr_puzz)
        num_zeros = np.size(curr_puzz) - np.count_nonzero(curr_puzz)
        
        #print('Episode: %d, Num_iters: %d, Num elements remaining: %d' % (e, num_iters, num_zeros))
        num_iters += 1
        
        #input('Press enter to continue\n\n ------ END OF TIMESTEP -------\n\n')

        
        if done:          
            print("\nFinished episode: {}/{}".format(e, num_episodes))
            is_done = True
            break
    
    #print('Finished working on the puzzle!')
    
    # train the agent with the experience of the episode
    # if num_iters >= 63:
    #     agent.replay(64)
    # else:
    #     agent.replay(num_iters-1)
    
    scores.append(reward_for_ep)
    eps_history.append(agent.epsilon)
    steps_array.append(num_iters)
    
    avg_score = np.mean(scores[-100:])
    ep_info = 'Epsiode: %d, score: %0.3f, avg score (last 100 episodes): %0.3f, epsilon: %0.3f, steps: %d' % (e, reward_for_ep, avg_score, agent.epsilon, num_iters)
    print(ep_info)
    
    
    # Okay. My goal is to test if I can save the models properly. 
    # Both the custom saving (every nth iteration)
    # And automatic saving (save when it gets a new highscore)
    
    # Metrics tracking 
    train_metrics.append({'episode': e, 'reward': reward_for_ep, 'num_iters': num_iters,
                    'avg_reward': reward_for_ep/(num_iters), 'epsilon': agent.epsilon, 
                      'puzzle_num': puzz_counter-1})
    
    

    # Step 1. Saving best model
    if (avg_score > best_score):
        if not load_checkpoint: 
            agent.save_models_best()
        best_score = avg_score
        with open('models/bst_chkpts/info.txt', 'a') as file:
            file.write(ep_info + ' best score: ' + str(best_score) + '\n')
            
            
    # Step 2: Regularly saving checkpoints and testing best model
    if (e % chk_freq == 0):
        pd.DataFrame(train_metrics).to_csv('data/train_metrics.csv', index=None)
        helper.plot_learning_curve(scores, eps_history, figure_file)
        agent.save_models_regular()
        with open('models/reg_chkpts/info.txt', 'a') as file:
            file.write(ep_info + '\n')
            
            
        # TESTING CURRENT MODEL: 
        # Choose a puzzle randomly that was not seen up to now 

        if (enable_tests == True):
            print('Testing using only the model!')
            
            test_puzz = np.random.randint(puzz_counter, 1000000) # Test randomly on a puzzle we haven't seen yet
            test_env.base = puzzles[test_puzz][0] # Unfinished puzzle
            test_env.sol = puzzles[test_puzz][1]  # Solution (used to inform reward function)
    
            state = test_env.reset()
            test_reward_for_ep = 0
            
            last_guess = None 
            last_guess_repeated_count = 0
            got_stuck = False
            
            test_num_iters = 0
            while True:
                action = agent.test_act(state.reshape(1,-1)) # Use a different action function that acts using model
                
                #print('Test iter: %d, action: %d' % (test_num_iters, action))
    
                if (last_guess == action):
                    last_guess_repeated_count += 1
                else: 
                    last_guess = action 
                    last_guess_repeated_count = 0
                
                if (last_guess_repeated_count > 10):
                    print('Model got in an infinite loop! Breaking!')
                    got_stuck = True
                    break
                
                
                next_state, reward, done, _ = test_env.step(action)
                test_reward_for_ep += reward
                
                state = next_state
                is_solved = np.array_equal(test_env.sol, state)
                
                test_num_iters += 1
                if ((done is True) or (test_num_iters > 1000)):
                    if ((done is True) and (is_solved is True)):
                        print('Model solved puzzle #: %d' % (test_puzz))            
                    elif (done is False):
                        print('Model failed to solve puzzle %d in <1k steps!' % (test_puzz))
                    break
            
            test_metrics.append({'num_train_episodes': e, 'reward': test_reward_for_ep,
                                  'num_iters': test_num_iters, 'avg_reward': test_reward_for_ep/(test_num_iters+1),
                                  'puzzle_num': test_puzz, 'got_stuck': got_stuck*1})
            pd.DataFrame(test_metrics).to_csv('data/test_metrics.csv', index=None)
