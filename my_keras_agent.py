from tensorflow import keras
from collections import deque
import numpy as np
import random


class SudokuAgent:
    def __init__(self):
        self.state_size = 9*9
        self.action_size = 9*9*9
        self.memory = deque(maxlen=10000) # create a queue of max length 1000. new entries after 1000 will replace old entries
        self.gamma = 0.95    # discount rate
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999
        self.learning_rate = 0.005
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = keras.Sequential()
        model.add(keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(keras.layers.Dense(24, activation='relu'))
        model.add(keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, status):
        self.memory.append((state, action, reward, next_state, status))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size=64, sample = True):

        if sample:
            batch = random.sample(self.memory, batch_size)
        else:
            batch = self.memory
            
        # generate batch
        # want data to have size (batch_size, 729), target to have size (batch_size, 729)
        batch_data = np.empty((batch_size, 81)) 
        batch_target = np.empty((batch_size, 729))

        for state, action, reward, next_state, done in batch:
            batch_data = np.append(batch_data, state, axis=0)
            target = reward
            if not done: # if not the terminal state
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target 
            batch_target = np.append(batch_target, target_f, axis = 0)

            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay