from tensorflow import keras
import tensorflow as tf
from collections import deque
import random
import gym
import sudoku_env
import numpy as np
import my_keras_agent
import os
import matplotlib.pyplot as plt
import pandas as pd


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def arr_index_to_action(index):
    """
    Maps an index 0-728 to an action: eg, put 3 in row 5, col 2
    The index will be the output of the NN's predict function because it takes an
    argmax of 729 layers
    """
    
    num_to_fill = index//81 + 1 # +1 as numbers are 1-9, not 0-8
    
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

def correct_boxes(init_state, final_state, sol):
    """
    Given the initial state, final state and solution, calculate the number of boxes the agent got corret
    at the end of an episode
    """
    num_init = np.count_nonzero(init_state)
    num_same = np.equal(final_state,sol).sum()
    
    return num_same - num_init #subtract out the numbers given at the start of the puzzle

load = True # indicate if you want to load an existing model
# model is saved in folder /my_model

env = gym.make('Sudoku-v0')
puzzles = np.load('puzzles.npy')
agent = my_keras_agent.SudokuAgent()

# This is how we pass in the puzzle to be solved
if load:
    params = np.loadtxt('agent_data.txt')
    agent.epsilon = params[0]
    puzz_counter = int(params[1]) + 1
    agent.model = keras.models.load_model("my_model")
    
    avg_reward_array = np.load('avg_reward.npy')
    correct_boxes_array = np.load('num_corr_boxes.npy')
    

else:
    puzz_counter = 0

env.base = puzzles[puzz_counter][0] # Unfinished puzzle
env.sol = puzzles[puzz_counter][1]  # Solution (used to inform reward function)


# training loop

if train is True:
    
    episodes = 1000
    done = True
    avg_reward_array = []
    correct_boxes_array = []

    # Iterate the game
    for e in range(episodes):

        if done:
            init_state = puzzles[puzz_counter][0] # Unfinished puzzle
            solution = puzzles[puzz_counter][1] # Solution
            env.base = np.copy(init_state)
            env.sol = np.copy(solution)
            init_states_index = get_init_states_index(puzzles[puzz_counter][0])
            env.init_states_index = init_states_index
            puzz_counter += 1

        state = env.reset()
        reward_for_ep = 0

        for num_iters in range(1,10001):

            # turn this on if you want to render
            # env.render()

            # Decide action
            action_index_flat = agent.act(state.reshape(1,-1))
            action = arr_index_to_action(action_index_flat)

            # Advance the game
            next_state, reward, done, _ = env.step(action)
            reward_for_ep += reward

            # memorize the previous state, action, reward, and done
            agent.memorize(state.reshape(1,-1), action_index_flat, reward, next_state.reshape(1,-1), done)

            if done:
                break

            # make next_state the new current state for the next frame.
            state = np.copy(next_state)

        avg_reward = reward_for_ep/num_iters
        avg_reward_array.append(avg_reward)

        print("finished ep {:d} with average reward {:f} in {:d} iters\n".format(e, avg_reward, num_iters))

        num_correct_boxes = correct_boxes(init_state, state, solution)
        correct_boxes_array.append(num_correct_boxes)

        # train the agent with memory of rewards
        if num_iters >= 64:
            agent.replay(64)
        else:
            agent.replay(num_iters)

        # save progress periodically    
        if e !=0 and e % 100 == 0:
            agent.model.save("my_model")
            np.savetxt('agent_data.txt',np.array([agent.epsilon, puzz_counter]))
            num_corr_boxes_save = np.array(correct_boxes_array)
            avg_reward_save = np.array(avg_reward_array)
            np.save('num_corr_boxes.npy', num_corr_boxes_save)
            np.save('avg_reward.npy', avg_reward_save)

# Plot average reward per episode during training

avg_reward_norm = avg_reward_array - np.mean(avg_reward_array)
episodes_array = np.arange(0, 901)
bin_size = 20

train =  pd.DataFrame(data=[episodes_array, avg_reward_norm]).T
train.columns = ['episode', 'avg_reward']
train['group'] = ((1/bin_size)*train['episode']).astype(int)

temp = train[['avg_reward', 'group']].groupby(['group']).mean()
temp['group'] = temp.index * bin_size
temp.reset_index(drop=True)

f = plt.figure(figsize=(5,4))
plt.plot(temp['group']/9, temp['avg_reward'])

ax=plt.gca()
plt.xlim([-2,102])
plt.ylabel('Normalized Average Reward',fontsize=15)
#f.savefig("avg_reward.jpg",dpi=300)

if evaluate is True:

    num_correct_boxes = []

    for i in range(-1,-101, -1):
        
        # choose last 100 puzzles in the dataset for evaluation

        init_state = puzzles[i][0] # Unfinished puzzle
        solution = puzzles[i][1] # Solution
        env.base = np.copy(init_state)
        env.sol = np.copy(solution)
        init_states_index = get_init_states_index(puzzles_for_test[i][0])
        env.init_states_index = init_states_index
        agent.epsilon = 0
        state = env.reset()

        last_action_count = 0
        num_iters = 0
        
        to_break = False
        
        while True:
            action_index_flat = agent.act(state.reshape(1,-1))
            action = arr_index_to_action(action_index_flat)

            if num_iters != 0 and np.array_equal(action, last_action):
                last_action_count += 1
            else:
                last_action_count = 0

            next_state, reward, status, _ = env.step(action)
            state = np.copy(next_state)

            last_action = np.copy(action)
            print(action)

            if last_action_count == 100 :
                print('stuck in infinite loop')
                to_break = True
            elif np.array_equal(state, solution):
                print("finished")
                to_break= True
            
            if to_break:
                num_correct_boxes_ = np.equal(state, env.sol).sum()
                num_correct_boxes.append(num_correct_boxes_)
                to_break = False
                break
                
            num_iters += 1

# Plot number of correct boxes during evaluation iterations
f = plt.figure()
plt.plot(num_correct_boxes)
plt.ylabel('Number of correct boxes')
plt.xlabel('Iterations')
plt.title('Correct boxes vs iteration')
plt.show()

