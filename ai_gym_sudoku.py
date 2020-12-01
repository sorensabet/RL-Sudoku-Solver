# import numpy as np
# import pandas as pd 
import sys
import gym
import gym_sudoku
import numpy as np 
import helper
import soren_solver
import time
import math
import random
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K

# Deep Q-learning Agent
# Code for DQQ from: https://github.com/keon/deep-q-learning/blob/master/ddqn.py
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=4000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.90  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    """Huber loss for Q Learning
    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            rand_act = mapping[np.random.randint(0,729)]
            row = rand_act[0]
            col = rand_act[1]
            val = rand_act[2]  
            return row, col, val
        
        act_values = mapping[np.argmax(self.model.predict(np.array([state.reshape(-1)])))]
        row = act_values[0]
        col = act_values[0]
        val = act_values[0]
        
        #print('Agent Action! %s' % str(act_values))
        return row, col, val
    
    # DDQN Code
    def replay(self, batch_size):
        print('Fitting model!')

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            np_state = tf.convert_to_tensor(np.array([state.reshape(-1)]))
            np_next_state = tf.convert_to_tensor(np.array([next_state.reshape(-1)]))
            
            target = self.model.predict(np_state)
            action_idx = np.where((mapping  == action).all(axis=1))

            if done: 
                target[0][action_idx] = reward
            else:
                t = self.target_model.predict(np_next_state)[0]
                target[0][action_idx] = reward + self.gamma * np.amax(t)
            
            self.model.fit(np_state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)



# Load data
puzzles = np.load('puzzles.npy')
mapping = np.load('729_mapping.npy')
num_let_map = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
curr_puzz = puzzles[0][0]

# Model parameters
num_episodes = 10000

# Initialize environment
env = gym.make('Sudoku-v0')
env.base = puzzles[0][0] # Unfinished puzzle
env.sol = puzzles[0][1]  # Solution (used to inform reward function)

state_size = 81
action_size = len(mapping)
agent = DQNAgent(state_size, action_size)
done=False
batch_size=64

best_score = None

puzz_num = 0
for episode in range(num_episodes):
        
    state = env.reset()
    score = 0

    start_grid = env.grid
    is_done = False

    for time_step in range(500):
    
        
        # Decide action
        action = agent.act(state)
            
        next_state, reward, done, _ = env.step(action) # take a random action
        score += reward
        
        # I need to check if the puzzle is finished (if next_state is complete)
        # So that if it is complete, the puzzle doesn't keep going on endlessly
        is_done = helper.check_solution_auto(env.grid, env.sol)
        if (is_done is True):
            break
        
        # I'm also going to put a check if the puzzle is still solvable after a legal move is made 
        # If the puzzle is not solvable, it should be penalized very heavily and the episode should end
        
        #print(next_state)
        #input('Batman')
        
        # next_state = step_res[0] # An environment specific object representing our observation (e.g. Pixel data from Camera, Joint Angles, Joint velocities, Board state) 
        # reward = step_res[1] # Amount of reward achieved by previous action 
        # done = step_res[2] # Whether it is time to reset the environment
        # info   = step_res[3] # Diagnostic info for debugging
        
        agent.memorize(state, action, reward, next_state, done)
        state = next_state
        
        print('\tEpisode: %d-%d, score: %s, best_score: %s'% (episode, time_step, str(score), str(best_score)))
        
        act_row = num_let_map[action[0]-1]
        act_col = action[1] 
        act_val = action[2] 
            
        #env.render()
        # print('Last action: [%s,%d] ==> %d \n\n\n\n\n' % (act_row, act_col, act_val))
        # input('Batman ')
        
        if done:
            agent.update_target_model()
            print("episode: {}/{}, score: {}, e: {:.2}".format(episode, num_episodes, time_step, agent.epsilon))
            break
    
    if (best_score is None):
        best_score = score
    else: 
        if (score > best_score):
            best_score = score
    
    print('Episode: %d, score: %d, best_score: %d, epsilon: %0.3f'% (episode,score, best_score, agent.epsilon))
    env.render()
    
    
    if (is_done is True): 
        print('COMPLETED A PUZZLE CORRECTLY!')
        with open('logs.txt', 'a') as file:
            file.write('Completed puzzle ' + str(puzz_num) + ' correctly!')
        
        puzz_num += 1
        env.base = puzzles[puzz_num][0] # Unfinished puzzle
        env.sol = puzzles[puzz_num][1] 
    
    if (start_grid == env.grid).all():
        print('No changes detected in episode, changing game board!')
        with open('logs.txt', 'a') as file:
            file.write('Could not solve puzzle ' + str(puzz_num) + ' correctly.')
        
        puzz_num += 1
        env.base = puzzles[puzz_num][0] # Unfinished puzzle
        env.sol = puzzles[puzz_num][1] 
    
    agent.replay(batch_size)
       

    
        # Reward of 1 indicates legal move, reward of 2 indicates correct move
        #if (step_res[1] == 1 or step_res[1] == 2):
        # print('Time Step: %d, Reward: %d' % (i, step_res[1]))
        # input('Press enter to continue!')    
        # else: 
        #     print('Time Step: %d, Reward: %d' % (i, step_res[1]))
        # print('\n\n\n')
        # input('Press enter to continue!')
    
env.close()
