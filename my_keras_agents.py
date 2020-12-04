import random
import numpy as np
from tensorflow import keras
from collections import deque

TEST = False

class SudokuAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.good_memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.999  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.model = self._build_model()
    
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
