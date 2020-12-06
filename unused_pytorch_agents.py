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

class LinearDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims, name='DQN', chkpt_dir='checkpoints'):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        
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
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent():
    def __init__(self, input_dims, n_actions, lr=0.0001, gamma=0.99,
                 epsilon=1.0, eps_dec=1e-7, eps_min=0.10, right_guess=0.25):
        self.lr = lr 
        self.input_dims = input_dims 
        self.n_actions = n_actions 
        self.gamma = gamma 
        self.epsilon = epsilon
        self.right_guess = right_guess # When making a random guess, the % of times it should use a deterministic solver
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.action_space = [i for i in range(self.n_actions)]
        
        self.Q = LinearDeepQNetwork(self.lr, self.n_actions, self.input_dims, 'checkpoint.pt', 'checkpoints')
    
    def act(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(observation, dtype=T.float).to(self.Q.device)
            actions = self.Q.forward(state)
            action = T.argmax(actions).item()
        else:
            action = get_modified_random_action(observation, self.action_space)
        return action 

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def learn(self, state, action, reward, state_):
        self.Q.optimizer.zero_grad()
        states = T.tensor(state, dtype=T.float).to(self.Q.device)
        actions = T.tensor(action).to(self.Q.device)
        rewards = T.tensor(reward).to(self.Q.device)
        states_ = T.tensor(state_, dtype=T.float).to(self.Q.device)
    
        q_pred = self.Q.forward(states)[actions]
        q_next = self.Q.forward(states_).max()
        q_target = reward + self.gamma*q_next
        
        loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
        loss.backward()
        self.Q.optimizer.step()
        self.decrement_epsilon()

class DQNAgent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, 
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7, 
                 replace=10000, algo=None, env_name=None, chkpt_dir='tmp/dqn', right_guess=0.25):
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
        self.chkpt_dir = chkpt_dir 
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0
        self.batch_size = batch_size
        self.right_guess = right_guess # When making a random guess, the % of times it should use a deterministic solver

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        
        self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                                   input_dims=self.input_dims,
                                   name=self.env_name + '_' + self.algo + '_q_eval',
                                   chkpt_dir=self.chkpt_dir)
   
        # We use this for calculating activations of resulting state-action 
        self.q_next = DeepQNetwork(self.lr, self.n_actions,
                                   input_dims=self.input_dims,
                                   name=self.env_name + '_' + self.algo + '_q_next',
                                   chkpt_dir=self.chkpt_dir)        
        
    def act(self, observation):
        # This part might need some work
        if np.random.random() > self.epsilon: 
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = get_modified_random_action(observation, self.action_space)
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
    
    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()
    
    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()
        
    def learn(self): 
        if self.memory.mem_cntr < self.batch_size:
            return 
        
        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()
        
        states, actions, rewards, states_, dones = self.sample_memory()
        
        indices = np.arange(self.batch_size)
        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_).max(dim=1)[0]
        q_next[T.gt(dones,0)] = 0.0 
        q_target = rewards + self.gamma*q_next
        
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        
        self.decrement_epsilon()

class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims, name='DQN', chkpt_dir='checkpoints'):
        super(DuelingDeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        
        self.fc1 = nn.Linear(input_dims, 64)
        self.fc2 = nn.Linear(64, 32)
        
        self.V = nn.Linear(32, 1)
        self.A = nn.Linear(32, n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        # layer1 = F.relu(self.fc1(state))
        # actions = self.fc2(layer1)
        
        layer1 = F.relu(self.fc1(state))
        layer2 = F.relu(self.fc2(layer1))
        
        V = self.V(layer2)
        A = self.A(layer2)
        return V, A
    
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class DuelingDQNAgent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, 
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7, 
                 replace=10000, algo=None, env_name=None, chkpt_dir='tmp/dqn', right_guess=0.25):
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
        self.chkpt_dir = chkpt_dir 
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0
        self.batch_size = batch_size
        self.right_guess = right_guess # When making a random guess, the % of times it should use a deterministic solver
        
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        
        self.q_eval = DuelingDeepQNetwork(lr, n_actions, input_dims)(self.lr, self.n_actions,
                                   input_dims=self.input_dims,
                                   name=self.env_name + '_' + self.algo + '_q_eval',
                                   chkpt_dir=self.chkpt_dir)
   
        # We use this for calculating activations of resulting state-action 
        self.q_next = DuelingDeepQNetwork(self.lr, self.n_actions,
                                   input_dims=self.input_dims,
                                   name=self.env_name + '_' + self.algo + '_q_next',
                                   chkpt_dir=self.chkpt_dir)        
        
    def act(self, observation):
        # This part might need some work
        if np.random.random() > self.epsilon: 
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            _, advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()
        else:
            action = get_modified_random_action(observation, self.action_space)
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
    
    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()
    
    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()
        
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return 
        
        self.q_eval.optimizer.zero_grad()
        
        self.replace_target_network()
        
        states, actions, rewards, states_, dones = self.sample_memory()
        
        indices = np.arange(self.batch_size)
       
        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)
        
        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_,(A_s_ - A_s_.mean(dim=1, keepdim=True))).max(dim=1)[0]
        
        q_next[T.gt(dones,0)] = 0.0 
        q_target = rewards + self.gamma*q_next
        
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()     
