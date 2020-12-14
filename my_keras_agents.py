import random
import numpy as np
from tensorflow import keras
from collections import deque

TEST = False

class SudokuAgent:
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, algo=None, env_name=None, ):
        self.state_size = input_dims
        self.action_size = n_actions
        self.memory = deque(maxlen=5000)
        self.good_memory = deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        self.model = self._build_model()

        self.algo = algo
        self.env_name = env_name
        self.batch_size = batch_size
    
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = keras.Sequential()
        model.add(keras.layers.Dense(48, input_dim=self.state_size, activation='relu'))
        model.add(keras.layers.Dense(48, activation='relu'))
        model.add(keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon and not TEST:
            # We want a weighted action so that the model is equally likely 
            # to sample from actions 
            
            # Experiment, try with two actions (move left and right)
            # Where left takes you to the next empty cell along rows
            #       right takes you to the previous empty cell along rows
            action = random.randrange(self.action_size)
            #if DEBUG: print('DEBUG: Random action', action)
            return action
        else:
            qsa = self.model.predict(state)
            #if DEBUG: print('DEBUG: qsa', qsa)
            action = np.argmax(qsa[0])
            #if DEBUG: print('DEBUG: action', action)
            return action
        
    def test_act(self, state):
            qsa = self.model.predict(state)
            #if DEBUG: print('DEBUG: qsa', qsa)
            action = np.argmax(qsa[0])
            #if DEBUG: print('DEBUG: action', action)
            return action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target 
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
