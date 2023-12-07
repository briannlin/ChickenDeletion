import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
import json
import math
import random



class Agent():
    def __init__(self):
        self.value_table = np.zeros((9800, 5))
        self.magnitude_table = np.ones((9800, 5))
        self.nn_input = np.zeros((9800, 4))
        self.prev_state_space = [2, 9, 0, 0]
        self.same_count = 0
        self.prev_row = 0
        self.prev_column = 0
        self.prev_info = {"entities":[{"yaw":270.0,"x":0.5,"y":4.0,"z":0.5,"pitch":30.0,"id":"3637d6bc-9d46-3e7a-8326-bc957d353e98","motionX":0.0,"motionY":-0.0784000015258789,"motionZ":0.0,"life":20.0,"name":"Chicken Deleter"},{"yaw":0.0,"x":3.5,"y":4.0,"z":5.5,"pitch":0.0,"id":"3670ddce-9e97-4542-9278-e7020baaf2b3","motionX":0.0,"motionY":-0.027635999999999997,"motionZ":0.0,"life":4.0,"name":"Chicken"}],"LineOfSight":{"hitType":"block","x":3.30597412875441,"y":4.0,"z":0.5,"type":"grass","prop_snowy":False,"inRange":True,"distance":3.2400448322296143},"DistanceTravelled":0,"TimeAlive":30,"MobsKilled":0,"PlayersKilled":0,"DamageTaken":0,"DamageDealt":0,"Life":20.0,"Score":0,"Food":20,"XP":0,"IsAlive":True,"Air":300,"Name":"Chicken Deleter","XPos":0.5,"YPos":4.0,"ZPos":0.5,"Pitch":30.0,"Yaw":270.0,"WorldTime":5000,"TotalTime":36}
    
    def action_selection(self, state_space):
        action_dict = {0: "move ", 1: "turn ", 2: "attack ", 3: "pitch ", 4: "jump "}
        index = self.state_space_index(state_space)
        column = np.argmax(self.value_table, axis=1)[index]
        self.prev_column = column
        return action_dict.get(column)

    def magnitude_selection(self, state_space, action):       
        column_index = {"move ": 0, "turn ": 1, "attack ": 2, "pitch ": 3, "jump ": 4}
        index = self.state_space_index(state_space)
        column = column_index.get(action)
        magnitude = self.magnitude_table[index][column]
        if magnitude % 1 == 0:
            return int(magnitude)
        else:
            return magnitude
        

    def state_space_function(self, info):
        try:
            info = json.loads(info)
            self.prev_info = info
        except:
            print(f"exception thrown in state space function")
            info = self.prev_info
        # agent info
        try:
            pitch = round((info.get("Pitch")) / 15)
        except:
            pitch = 0
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
        angle = math.atan2((chicken_z - agent_z), (chicken_x - agent_x))
        angle = (math.degrees(angle) + 360) % 360
        yaw = info.get("Yaw")
        offset = 90 - yaw
        angle = 180 - (angle + offset) % 360

        chicken_diff = round(chicken_to_agent - chicken_to_hit_block, 1) * 10
        if chicken_diff < 0:
            chicken_diff = 0
        if chicken_to_agent > 10:
            chicken_to_agent = 10
    
        if abs(angle) <= 15:
            angle_state = 2
        elif angle < 0:
            angle_state = 1
        else:
            angle_state = 0
        if line_of_sight.get("hitType") == "entity" and line_of_sight.get("inRange"):
            angle_state = 3

        self.prev_state_space = [pitch, round(chicken_to_agent), angle_state, chicken_diff]
        return self.prev_state_space
        
    
    def state_space_index(self, state_space):
        index = int(state_space[0] + state_space[1] * 7 + state_space[2] * 70 + state_space[3] * 280)
        self.prev_row = index
        return index
    
    def reward_adjustment(self, reward):
        if reward < np.argmax(self.value_table, axis=1)[self.prev_row]:
            self.value_table[self.prev_row][self.prev_column] - 5
            self.train_action_nn


class Reward():
    def __init__(self):
        self.previous_dist = 15
        self.mobs_killed = 0
    
    def calculate_reward(self, info):
        try:
            reward = 0
            info = json.loads(info)
            if info.get("MobsKilled") > self.mobs_killed:
                self.mobs_killed += 1
                return 10000
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
            angle = math.atan2((chicken_z - agent_z), (chicken_x - agent_x))
            angle = (math.degrees(angle) + 360) % 360
            yaw = info.get("Yaw")
            offset = 90 - yaw
            angle = abs(180 - (angle + offset) % 360)

            reward += 20 - chicken_to_agent
            if line_of_sight.get("hitType") == "entity" and line_of_sight.get("inRange"):
                reward += 250
            elif chicken_to_hit_block < 0.25:
                reward += 175
            else: 
                reward -= 150
            if angle < 60:
                reward += (60-angle) * 1.5
            else:
                reward -= angle * 15
            return reward
        
        except:
            return -600


        

        