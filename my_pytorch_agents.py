# Code from: https://www.udemy.com/course/deep-q-learning-from-paper-to-code/learn/lecture/17009520#overview
# Please buy the course to support the instructor!

import os 
import gym
import numpy as np
import pandas as pd 

import torch as T 
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F

import helper
import soren_solver

# Function to give accurate action some percentage of the time when epsilon is high 
def get_modified_random_action(state, right_guess_percent, action_space):
    """
    Given the state of the board as well as the current position, determines if a human 
    Could solve the cell and returns this value x % of the time
    """
    # For development, just use a deterministic action, then put the logic inside if statements
    
    if np.random.random() < right_guess_percent: 
        #print('Randomly selected action - DETERMINISTIC!')
        grid = np.array(state[0][:81]).reshape((9,9))
        row = state[0][81]
        col = state[0][82]
        
        can_solve, val = soren_solver.check_human_can_solve_cell(row, col, grid)
        
        #print('Guessed a deterministic action!')
        #print('can_solve: %s' % str(can_solve))
        
        # If the current cell can be solved, return the correct value 
        # Otherwise, move right
        if can_solve:
            return val - 1
        else: 
            return 10
    else:
        #print('Randomly selected action - RANDOM')
        action = np.random.choice(action_space)
    
    return action

class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims, name='DQN', bst_chkpt_dir='bst_chkpts', reg_chkpt_dir='reg_chkpts'):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = bst_chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        
        self.reg_checkpoint_dir = reg_chkpt_dir
        self.reg_checkpoint_file = os.path.join(self.reg_checkpoint_dir, name)
         
        self.fc1 = nn.Linear(input_dims, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        # layer1 = F.relu(self.fc1(state))
        # actions = self.fc2(layer1)
        
        layer1 = F.relu(self.fc1(state))
        layer2 = F.relu(self.fc2(layer1))
        actions = self.fc3(layer2)
        return actions 
    
    def save_checkpoint(self):
        print('... saving best checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        print('... loading best checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
        
    def save_reg_checkpoint(self):
        print('... saving reg checkpoint ...')
        T.save(self.state_dict(), self.reg_checkpoint_file)
        
    def load_reg_checkpoint(self):
        print('... loading reg checkpoint ...')
        self.load_state_dict(T.load(self.reg_checkpoint_file))
        
class DDQNAgent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, 
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7, 
                 replace=10000, algo=None, env_name=None, 
                 bst_chkpt_dir='models/bst_checkpts', 
                 reg_chkpt_dir='models/reg_chkpts',
                 right_guess=0.25):
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions 
        self.input_dims = input_dims
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace 
        self.algo = algo
        self.env_name = env_name
        self.bst_chkpt_dir = bst_chkpt_dir 
        self.reg_chkpt_dir = reg_chkpt_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0
        self.batch_size = batch_size
        self.right_guess = right_guess # When making a random guess, the % of times it should use a deterministic solver
        
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        
        self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                                   input_dims=self.input_dims,
                                   name=self.env_name + '_' + self.algo + '_q_eval',
                                   bst_chkpt_dir=self.bst_chkpt_dir,
                                   reg_chkpt_dir=self.reg_chkpt_dir)
   
        # We use this for calculating activations of resulting state-action 
        self.q_next = DeepQNetwork(self.lr, self.n_actions,
                                   input_dims=self.input_dims,
                                   name=self.env_name + '_' + self.algo + '_q_next',
                                   bst_chkpt_dir=self.bst_chkpt_dir,
                                   reg_chkpt_dir=self.reg_chkpt_dir)        
        
    def act(self, observation):
        # This part might need some work
        if np.random.random() > self.epsilon: 
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = get_modified_random_action(observation, self.right_guess, self.action_space)
        return action 

    def test_act(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
        actions = self.q_eval.forward(state)
        action = T.argmax(actions).item()
        return action 
    
    def store_transition(self, state, action, reward, state_, done): 
        self.memory.store_transition(state, action, reward, state_, done)
        
    def sample_memory(self):
        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)     
            
        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)
        
        return states, actions, rewards, states_, dones
    
    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
            
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min
    
    def save_models_best(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()
    
    def load_models_best(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()
        
    def save_models_regular(self):
        self.q_eval.save_reg_checkpoint()
        self.q_next.save_reg_checkpoint()
        
    def load_models_regular(self):
        self.q_eval.load_reg_checkpoint()
        self.q_next.load_reg_checkpoint()
        
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return 
        
        self.q_eval.optimizer.zero_grad()
        
        self.replace_target_network()
        
        states, actions, rewards, states_, dones = self.sample_memory()
        
        indices = np.arange(self.batch_size)
        
        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_)
        q_next[T.gt(dones,0)] = 0.0 

        q_eval = self.q_eval.forward(states_)
        
        max_actions = T.argmax(q_eval, dim=1)
        
        q_target = rewards + self.gamma*q_next[indices, max_actions]
        
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()
  

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size    # Maximum size of memory
        self.mem_cntr = 0           # Position of last stored memory
        self.state_memory = np.zeros((self.mem_size, input_shape),dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape),dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)
        
    def store_transition(self, state, action, reward, state_, done):
        # Store the memories in the position of the first unoccupied memory
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action 
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done 
        self.mem_cntr += 1
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]
        
        return states, actions, rewards, states_, dones 
           
       
