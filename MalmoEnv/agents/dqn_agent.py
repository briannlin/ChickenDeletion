from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

COMMANDS_LIST = ['move 1', 'pitch 1', 'pitch -1', 'turn 1', 'turn -1', 'attack 1',
                 'attack 0', 'turn 0', 'move 0']
COMMANDS_DICT = {index: obj for index, obj in enumerate(COMMANDS_LIST)}

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, experience):
        """Save a transition"""
        self.memory.append(experience)

    def sample(self, batch_size, state_size):
        if batch_size > len(self.memory):
            experiences = [random.sample(self.memory, 1)[0] for _ in range(batch_size)]
        else:
            experiences = random.sample(self.memory, batch_size)

        states, next_states = torch.zeros((batch_size, state_size)), torch.zeros((batch_size, state_size))
        actions, rewards, dones = torch.zeros((batch_size, 1)), torch.zeros((batch_size, 1)), torch.zeros((batch_size, 1))
        for i, experience in enumerate(experiences):
            states[i] = torch.FloatTensor(experience[0])
            actions[i] = experience[1]
            rewards[i] = experience[2]
            next_states[i] = torch.FloatTensor(experience[3])
            dones[i] = experience[4]
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(state_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action_size)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.batch_size = 128
        self.gamma = 0.99
        self.update_every = 4
        self.epsilon = 1.0
        self.epsilon_min = 0.15
        self.epsilon_decay = 0.997
        self.tau = 0.005
        self.lr = 1e-4

        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(10000)

    def act(self, state):
        if random.random() <= self.epsilon:
            # Random action with probability epsilon
            return np.random.choice(self.action_size)
        else:
            # Act according to local q-network - select the action with highest Q-value
            self.policy_net.eval()
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)
                out = self.policy_net(state)
        self.policy_net.train()

        optimal_action = torch.argmax(out).item()
        print(f"optimal action for state {state}: {COMMANDS_DICT.get(optimal_action)}")
        return optimal_action

    def remember(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))

    def replay(self, current_step):
        if len(self.memory) < self.batch_size:
            return

        # Perform training every few steps
        if current_step % self.update_every == 0:
            print("Updating/replaying...")
            # Minibatch from experience buffer, calculate loss between
            # policy's Q(s, a) and target's r + ɣ * maxQ(s’, a’)
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size, self.state_size)
            target_rewards = rewards + self.gamma * torch.max(self.target_net(next_states), dim=1)[0].unsqueeze(
                1) * (1 - dones)
            local_rewards = self.policy_net(states).gather(1, actions.long())

            criterion = nn.SmoothL1Loss()
            loss = criterion(local_rewards, target_rewards)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            self.optimizer.step()

            # Soft-update target network with policy network so it doesn't diverge too much
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (
                        1 - self.tau)
            self.target_net.load_state_dict(target_net_state_dict)


    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def save(self, filename='dqn_agent.pth'):
        checkpoint = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'batch_size': self.batch_size,
            'gamma': self.gamma,
            'update_every': self.update_every,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'tau': self.tau,
            'lr': self.lr,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'memory': self.memory,
        }
        torch.save(checkpoint, filename)
        print(f"Saved agent with epsilon: {self.epsilon} and memory length: {len(self.memory)}")

    def load(self, filename='dqn_agent.pth'):
        checkpoint = torch.load(filename)
        self.state_size = checkpoint['state_size']
        self.action_size = checkpoint['action_size']
        self.batch_size = checkpoint['batch_size']
        self.gamma = checkpoint['gamma']
        self.update_every = checkpoint['update_every']
        self.epsilon = checkpoint['epsilon']
        self.epsilon_min = checkpoint['epsilon_min']
        self.epsilon_decay = checkpoint['epsilon_decay']
        self.tau = checkpoint['tau']
        self.lr = checkpoint['lr']

        self.policy_net = DQN(self.state_size, self.action_size)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net = DQN(self.state_size, self.action_size)
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=checkpoint['lr'], amsgrad=True)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.memory = checkpoint['memory']

        print(f"Loaded agent with epsilon: {self.epsilon} and memory length: {len(self.memory)}")
