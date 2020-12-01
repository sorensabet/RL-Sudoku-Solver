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

def mapping_to_target_range(x, target_min=0, target_max=9) :
    x02 = K.tanh(x) + 1 # x in range(0,2)
    scale = ( target_max-target_min )/2.
    return  x02 * scale + target_min

# Deep Q-learning Agent
# Code for DQQ from: https://github.com/keon/deep-q-learning/blob/master/ddqn.py
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
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
        model.add(Dense(self.action_size, activation=mapping_to_target_range))
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
            row = random.randrange(9)
            col = random.randrange(9) 
            val = random.randrange(9)   
            #print('Random action')
            return row, col, val
        
        act_values = self.model.predict(np.array([state.reshape(-1)]))  
        row = math.floor(act_values[0][0])
        col = math.floor(act_values[0][1])
        val = math.floor(act_values[0][2])
        
        #print('Agent Action! %s' % str(act_values))
        return row, col, val

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            np_state = np.array([state.reshape(-1)])
            np_next_state = np.array([next_state.reshape(-1)])
            
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(np_next_state)[0]))
                

            target_f = self.model.predict(np_state)
            
            # print('Action:      %s' % str(action))
            # print('Reward:      %s' % str(reward))
            # print('Target:      %s' % str(target))
            # print('Target_f:    %s' % str(target_f[0]))
            
            # Sketchy solution that might not work, but I don't know what else to do:
            # Target_f: Future discounted reward
            # We want the agent to map the current state to future discounted reward 
            # Because we have 3 different actions, I will randomly assign one of the values to the new target array, 
            # I will assign all 3 values to the target array. 
            
            # This is the key line that I have to figure out
            target_f[0][random.randint(0,2)] = target
            
            # Old code that I don't really understand 
            # target_f[0][action] = target
            self.model.fit(np_state, target_f, epochs=1, verbose=0)
            
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)



# Load data
puzzles = np.load('puzzles.npy')
num_let_map = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
curr_puzz = puzzles[0][0]

# Model parameters
num_episodes = 5000

# Initialize environment
env = gym.make('Sudoku-v0')
env.base = puzzles[0][0] # Unfinished puzzle
env.sol = puzzles[0][1]  # Solution (used to inform reward function)

state_size = 81
action_size = 3
agent = DQNAgent(state_size, action_size)
done=False
batch_size=64

best_score = None

for i in range(num_episodes):
    
    state = env.reset()
    score = 0
        
    for t in range(200):
    
        #env.render()
        
        # Decide action
        action = agent.act(state)
                
        next_state, reward, done, _ = env.step(action) # take a random action
        score += reward
        # next_state = step_res[0] # An environment specific object representing our observation (e.g. Pixel data from Camera, Joint Angles, Joint velocities, Board state) 
        # reward = step_res[1] # Amount of reward achieved by previous action 
        # done = step_res[2] # Whether it is time to reset the environment
        # info   = step_res[3] # Diagnostic info for debugging
        
        
        # In the example, I think this is position, velocity, angle, and angular velocity
        # The reward is determined based on whether the cart stayed in the environment, and 
        # whether the pole tipped over or not. 
        # I think we can leave this out in ours
        #r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        #r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        #reward = r1 + r2

        agent.memorize(state, action, reward, next_state, done)
        state = next_state
        
        #print(action)
        
        if done:
            agent.update_target_model()
            print("episode: {}/{}, score: {}, e: {:.2}".format(i, num_episodes, t, agent.epsilon))
            break
    
    if (best_score is None):
        best_score = score
    else: 
        if (score > best_score):
            best_score = score
    
    print('Episode: %d, score: %d, best_score: %d, epsilon: %0.3f'% (i,score, best_score, agent.epsilon))
    
    agent.replay(batch_size)
       
        # act_row = num_let_map[env.last_action[0]]
        # act_col = env.last_action[1] + 1 # We add 1 here because grid columns are visualized from 1-9
        # act_val = env.last_action[2] + 1 # We add 1 here because the action space is from 0-8 and we want a value in 1-9
    
    
        # Reward of 1 indicates legal move, reward of 2 indicates correct move
        #if (step_res[1] == 1 or step_res[1] == 2):
        # print('Time Step: %d, Reward: %d' % (i, step_res[1]))
        # print('Last action: [%s,%d] ==> %d \n\n\n\n\n' % (act_row, act_col, act_val))
        # input('Press enter to continue!')    
        # else: 
        #     print('Time Step: %d, Reward: %d' % (i, step_res[1]))
        # print('\n\n\n')
        # input('Press enter to continue!')
    
env.close()
DQNAgent.save('last_trained_agent')
