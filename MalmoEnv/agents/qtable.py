import random

import numpy as np


COMMANDS_LIST = ['move 1', 'pitch 1', 'pitch -1', 'turn 1', 'turn -1', 'attack 1',
                  'attack 0', 'turn 0', 'move 0']
COMMANDS_DICT = {index: obj for index, obj in enumerate(COMMANDS_LIST)}

class QTable():
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.qtable = np.zeros((state_space, action_space))

        self.state_to_index = {}
        distances = np.arange(0, 5.5, 0.5).tolist()
        yaw_angles = np.arange(-90, 100, 10).tolist()
        pitch_angles = np.arange(-30, 40, 10).tolist()
        near_walls = [1.0, -1.0]
        index = 0
        for i, dist in enumerate(distances):
            for j, yaw_angle in enumerate(yaw_angles):
                for k, pitch_angle in enumerate(pitch_angles):
                    for l, near_wall in enumerate(near_walls):
                        self.state_to_index[(dist, yaw_angle, pitch_angle, near_wall)] = index
                        index += 1

        self.learning_rate = 0.7
        self.gamma = 0.95
        self.epsilon = 1.0
        self.min_epsilon = 0.05
        self.epsilon_decay = 0.995


    def epsilon_greedy_policy(self, state):
        # Randomly generate a number between 0 and 1
        state = self.state_to_index[tuple(state)]
        random_num = random.random()
        # if random_num > greater than epsilon --> exploitation
        if random_num > self.epsilon:
            # Take the action with the highest value given a state
            # np.argmax can be useful here
            action = np.argmax(self.qtable, axis=1)[state]
            print(f"optimal_action chosen for state {state}: {COMMANDS_DICT.get(action)}")
        # else --> exploration
        else:
            action = random.randint(0, self.action_space-1)

        return action

    def update(self, state, action, reward, new_state):
        tuple_state = state
        state = self.state_to_index[tuple(state)]
        new_state = self.state_to_index[tuple(new_state)]
        self.qtable[state][action] = self.qtable[state][action] + self.learning_rate * (
                    reward + self.gamma * np.max(np.array(self.qtable[new_state])) - self.qtable[state][action])
        print(f"Updated [{COMMANDS_DICT[action]}] at [{tuple_state}] to: ", self.qtable[state][action])

    def decay_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay