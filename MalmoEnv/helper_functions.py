import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
import json
import math
import random


"""
agent to chicken (22)
chicken to hitbox (22)
positive negative (2)
pitch (7)
"""

class Agent():
    def __init__(self):
        self.value_table = np.zeros((49000, 5))
        self.magnitude_table = np.zeros((49000, 5))
        self.nn_input = np.zeros((49000, 5))
        self.action_nn = MLPClassifier(hidden_layer_sizes=(5,10,8,5), activation='relu', solver='sgd', alpha=0.0001, max_iter=1000)
        self.prev_state_space = [0, 6, 7, 0, 1]
        self.same_count = 0
        self.prev_row = 0
        self.prev_column = 0
        self.prev_info = {"entities":[{"yaw":270.0,"x":0.5,"y":4.0,"z":0.5,"pitch":30.0,"id":"3637d6bc-9d46-3e7a-8326-bc957d353e98","motionX":0.0,"motionY":-0.0784000015258789,"motionZ":0.0,"life":20.0,"name":"Chicken Deleter"},{"yaw":0.0,"x":3.5,"y":4.0,"z":5.5,"pitch":0.0,"id":"3670ddce-9e97-4542-9278-e7020baaf2b3","motionX":0.0,"motionY":-0.027635999999999997,"motionZ":0.0,"life":4.0,"name":"Chicken"}],"LineOfSight":{"hitType":"block","x":3.30597412875441,"y":4.0,"z":0.5,"type":"grass","prop_snowy":False,"inRange":True,"distance":3.2400448322296143},"DistanceTravelled":0,"TimeAlive":30,"MobsKilled":0,"PlayersKilled":0,"DamageTaken":0,"DamageDealt":0,"Life":20.0,"Score":0,"Food":20,"XP":0,"IsAlive":True,"Air":300,"Name":"Chicken Deleter","XPos":0.5,"YPos":4.0,"ZPos":0.5,"Pitch":30.0,"Yaw":270.0,"WorldTime":5000,"TotalTime":36}

    def initialize_value_table(self):
        # turn for chicken not in line of sight and move if in line of sight
        for pitch in range(7):
            for chicken_to_agent in range(10):
                for chicken_to_hit_block in range(10):
                    for boolean in range(2):
                        for chicken_diff in range(35):
                            row = pitch + chicken_to_agent * 7 + chicken_to_hit_block * 70 + boolean * 700 + chicken_diff * 1400
                            self.magnitude_table[row][2] = 1
                            self.magnitude_table[row][4] = 1

                            if chicken_diff > 25:
                                self.value_table[row][0] += 100
                                self.value_table[row][1] -= 100
                            else:
                                self.value_table[row][1] += 100
                                self.value_table[row][0] -= 100
                                self.value_table[row][2] -= 100
                                self.value_table[row][3] -= 100
                                self.value_table[row][4] -= 100
                            
        
        self.train_action_nn()
    
    def train_action_nn(self):
        # initialize neural network input array
        for pitch in range(7):
            for chicken_to_agent in range(10):
                for chicken_to_hit_block in range(10):
                    for boolean in range(2):
                        for chicken_diff in range(35):
                            row = pitch + chicken_to_agent * 7 + chicken_to_hit_block * 70 + boolean * 700 + chicken_diff * 1400
                            self.nn_input[row][0] = pitch
                            self.nn_input[row][1] = chicken_to_agent
                            self.nn_input[row][2] = chicken_to_hit_block
                            self.nn_input[row][3] = boolean
                            self.nn_input[row][4] = chicken_diff
        classifier = MLPClassifier(hidden_layer_sizes=(5,10,8,5), activation='relu', solver='sgd', alpha=0.0001, max_iter=1000)
        classifier.fit(self.nn_input, np.argmax(self.value_table, axis=1))
        self.action_nn = classifier
    
    def action_selection_nn(self, state_space):
        action_dict = {0: "move ", 1: "turn ", 2: "attack ", 3: "pitch ", 4: "jump "}
        nn_return = self.action_nn.predict([state_space])
        #index = self.state_space_index(state_space)
        #column = np.argmax(self.value_table, axis=1)[index]
        #self.prev_column = column
        #return action_dict.get(column)
        self.prev_column = nn_return[0]
        return action_dict.get(nn_return[0])

    def magnitude_selection_nn(self, state_space, action):
        if action in ["attack ", "jump "]:
            return str(1)
        else:
            #column_index = {"move ": 0, "turn ": 1, "attack ": 2, "pitch ": 3, "jump ": 4}
            return round(5*(random.random() - 0.5), 1)

    def state_space_function(self, info):
        try:
            info = json.loads(info)
            self.prev_info = info
        except:
            print(f"exception thrown in state space function")
            info = self.prev_info
        # agent info
        pitch = round((info.get("Pitch")) / 15)
        agent_x = info.get("XPos")
        agent_z = info.get("ZPos")
        # chicken info
        chicken = info.get("entities")[1]
        chicken_x = chicken.get("x")
        chicken_z = chicken.get("z")
        # hit block info
        line_of_sight = info.get("LineOfSight")
        line_of_sight_x = line_of_sight.get("x")
        line_of_sight_z = line_of_sight.get("z")
        chicken_to_agent = math.sqrt((chicken_x - agent_x) ** 2 + (chicken_z - agent_z) ** 2)
        basis_x = line_of_sight_x - agent_x
        basis_z = line_of_sight_z - agent_z
        if basis_z == 0:
            perpendicular_z = 1
            perpendicular_x = 0
        else:
            perpendicular_z = -basis_x / basis_z
            perpendicular_x = 1
        change_of_basis_matrix_invert = np.linalg.inv(np.array([[basis_x, perpendicular_x], [basis_z, perpendicular_z]]))
        chicken_to_agent_new = np.matmul(change_of_basis_matrix_invert, np.array([[chicken_x - agent_x],[chicken_z - agent_z]]))
        print(f"chicken to agent new basis coordinates: {chicken_to_agent_new}")
        angle = math.atan2((chicken_to_agent_new[1][0]), (chicken_to_agent_new[0][0]))
        print(f"angle of chicken to agent: {angle}")
        chicken_to_hit_block = math.sqrt((chicken_x - line_of_sight_x) ** 2 + (chicken_z - line_of_sight_z) ** 2)
        chicken_diff = round(chicken_to_agent - chicken_to_hit_block, 1) * 10
        if chicken_diff < 0:
            chicken_diff = 0
        if chicken_to_agent > 10:
            chicken_to_agent = 10
        if chicken_to_hit_block > 10:
            chicken_to_hit_block = 10
        if angle < 0:
            boolean = 0
        else:
            boolean = 1
        self.prev_state_space = [pitch, round(chicken_to_agent), round(chicken_to_hit_block), boolean, chicken_diff]
        return self.prev_state_space
        
    
    def state_space_index(self, state_space):
        index = int(state_space[0] + state_space[1] * 7 + state_space[2] * 70 + state_space[3] * 700 + state_space[4] * 1400)
        self.prev_row = index
        return index
    
    def reward_adjustment(self, reward):
        if reward < np.argmax(self.value_table, axis=1)[self.prev_row]:
            self.value_table[self.prev_row][self.prev_column] - 5
            self.train_action_nn


class Reward():
    def __init__(self):
        self.previous_dist = 15
    
    def calculate_reward(self, info):
        try:
            reward = 0
            info = json.loads(info)
            if info.get("MobsKilled") == 1:
                return 5000
            # agent info
            agent_x = info.get("XPos")
            agent_z = info.get("ZPos")
            # chicken info
            chicken = info.get("entities")[1]
            chicken_x = chicken.get("x")
            chicken_z = chicken.get("z")
            # hit block info
            line_of_sight = info.get("LineOfSight")
            line_of_sight_x = line_of_sight.get("x")
            line_of_sight_z = line_of_sight.get("z")
            chicken_to_agent = math.sqrt((chicken_x - agent_x) ** 2 + (chicken_z - agent_z) ** 2)
            chicken_to_hit_block = math.sqrt((chicken_x - line_of_sight_x) ** 2 + (chicken_z - line_of_sight_z) ** 2)
            basis_x = line_of_sight_x - agent_x
            basis_z = line_of_sight_z - agent_z
            perpendicular_z = -basis_x / basis_z
            change_of_basis_matrix_invert = np.linalg.inv(np.array([[basis_x, 1], [basis_z, perpendicular_z]]))
            chicken_to_agent_new = np.matmul(change_of_basis_matrix_invert, np.array([[chicken_x - agent_x],[chicken_z - agent_z]]))
            angle = abs(math.atan2((chicken_to_agent_new[1][0]), (chicken_to_agent_new[0][0])))

            if chicken_to_agent - chicken_to_hit_block > 2.5:
                reward += 50
            elif chicken_to_hit_block < 0.25:
                reward += 175
            else: 
                reward -= 100
            if angle < 1:
                reward += (1-angle) * 20
            else:
                reward -= (angle) * 75
            return reward
        
        except:
            return 0


        

        