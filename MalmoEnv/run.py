# ------------------------------------------------------------------------------------------------
# Copyright (c) 2018 Microsoft Corporation
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

import malmoenv
import argparse
from pathlib import Path
import time
from PIL import Image
import random
from helper_functions import Reward, Agent
import numpy as np
import json


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='malmovnv test')
    parser.add_argument('--mission', type=str, default='missions/chicken_deletion.xml', help='the mission xml')
    parser.add_argument('--port', type=int, default=9000, help='the mission server port')
    parser.add_argument('--server', type=str, default='127.0.0.1', help='the mission server DNS or IP address')
    parser.add_argument('--port2', type=int, default=None, help="(Multi-agent) role N's mission port. Defaults to server port.")
    parser.add_argument('--server2', type=str, default=None, help="(Multi-agent) role N's server DNS or IP")
    parser.add_argument('--episodes', type=int, default=1, help='the number of resets to perform - default is 1')
    parser.add_argument('--episode', type=int, default=0, help='the start episode - default is 0')
    parser.add_argument('--role', type=int, default=0, help='the agent role - defaults to 0')
    parser.add_argument('--episodemaxsteps', type=int, default=0, help='max number of steps per episode')
    parser.add_argument('--saveimagesteps', type=int, default=0, help='save an image every N steps')
    parser.add_argument('--resync', type=int, default=0, help='exit and re-sync every N resets'
                                                              ' - default is 0 meaning never.')
    parser.add_argument('--experimentUniqueId', type=str, default='test1', help="the experiment's unique id.")
    args = parser.parse_args()
    if args.server2 is None:
        args.server2 = args.server

    xml = Path(args.mission).read_text()
    env = malmoenv.make()

    env.init(xml, args.port,
             server=args.server,
             server2=args.server2, port2=args.port2,
             role=args.role,
             exp_uid=args.experimentUniqueId,
             episode=args.episode, resync=args.resync,
             action_filter={"move", "turn", "attack", "pitch", "jump"})

    agent = Agent()
    column_index = {"move ": 0, "turn ": 1, "attack ": 2, "pitch ": 3, "jump ": 4}
    action_list = ["move ", "turn ", "attack "]
    epsilon = 0
    time_track_list = []

    with open('magnitude_trained.npy', 'rb') as f:
        agent.magnitude_table = np.load(f)
    with open('value_trained.npy', 'rb') as f:
        agent.value_table = np.load(f)
    
    

    for i in range(args.episodes):
        episode_time = 0
        
        print("reset " + str(i))
        obs = env.reset()

        steps = 0
        done = False
        rewards = Reward()
        state_space = [2, 9, 0, 0]
        previous_info = {"entities":[{"yaw":270.0,"x":0.5,"y":4.0,"z":0.5,"pitch":30.0,"id":"3637d6bc-9d46-3e7a-8326-bc957d353e98","motionX":0.0,"motionY":-0.0784000015258789,"motionZ":0.0,"life":20.0,"name":"Chicken Deleter"},{"yaw":0.0,"x":3.5,"y":4.0,"z":5.5,"pitch":0.0,"id":"3670ddce-9e97-4542-9278-e7020baaf2b3","motionX":0.0,"motionY":-0.027635999999999997,"motionZ":0.0,"life":4.0,"name":"Chicken"}],"LineOfSight":{"hitType":"block","x":3.30597412875441,"y":4.0,"z":0.5,"type":"grass","prop_snowy":False,"inRange":True,"distance":3.2400448322296143},"DistanceTravelled":0,"TimeAlive":30,"MobsKilled":0,"PlayersKilled":0,"DamageTaken":0,"DamageDealt":0,"Life":20.0,"Score":0,"Food":20,"XP":0,"IsAlive":True,"Air":300,"Name":"Chicken Deleter","XPos":0.5,"YPos":4.0,"ZPos":0.5,"Pitch":30.0,"Yaw":270.0,"WorldTime":5000,"TotalTime":36}
        while not done and (args.episodemaxsteps <= 0 or steps < args.episodemaxsteps):
            try:   
                state_space_index = agent.state_space_index(state_space)
                action_type = agent.action_selection(state_space)
                magnitude = agent.magnitude_selection(state_space, action_type)
                exploration = random.random()
                if exploration < epsilon:
                    action_type = random.choice(action_list)
                    magnitude = random.choice([-1, 1])

                action = action_type + str(magnitude)

                print(f"action that is taken: {action}")
                obs, reward, done, info = env.step(action)
                steps += 1
                custom_reward = rewards.calculate_reward(info)
                if info:
                    previous_info = info
            
                try:
                    loaded_info = json.loads(info)
                    episode_time = loaded_info.get("TotalTime")
                    print(f"episode time elapsed: {episode_time}")
                    if loaded_info.get("MobsKilled") == 5:
                        done = True
                except:
                    pass
                
                print("reward: " + str(custom_reward))
                print(info)
                new_state_space = agent.state_space_function(info)
                print(new_state_space[3])
                print(f"current state space: {new_state_space}")
                new_state_space_index = agent.state_space_index(new_state_space)
                print(f"current value function for state space: {agent.value_table[new_state_space_index]}")
                if action_type in ["move ", "turn ", "attack "]:
                    env.step(action_type + "0")
                
                
                if custom_reward > agent.value_table[state_space_index][column_index.get(action_type)]:
                    agent.value_table[state_space_index][column_index.get(action_type)] = custom_reward
                    print(f"new row values for value function: {agent.value_table[state_space_index]}")
                    if action_type == "turn ":
                        for pitch in range(7):
                            for chicken_to_agent in range(10):
                                for angle in range(4):
                                    for chicken_diff in range(35):
                                        if angle == state_space[2] and chicken_diff == state_space[3]:
                                            row = pitch + chicken_to_agent * 7 + angle * 70 + chicken_diff * 280
                                            if magnitude % 1 == 0:
                                                agent.magnitude_table[state_space_index][column_index.get(action_type)] = int(magnitude)
                                            else:
                                                agent.magnitude_table[state_space_index][column_index.get(action_type)] = float(magnitude)
                    else:
                        if magnitude % 1 == 0:
                            agent.magnitude_table[state_space_index][column_index.get(action_type)] = int(magnitude)
                        else:
                            agent.magnitude_table[state_space_index][column_index.get(action_type)] = magnitude
                
                state_space = new_state_space

                if args.saveimagesteps > 0 and steps % args.saveimagesteps == 0:
                    h, w, d = env.observation_space.shape
                    img = Image.fromarray(obs.reshape(h, w, d))
                    img.save('image' + str(args.role) + '_' + str(steps) + '.png')

                time.sleep(0.05)
            except:
                continue
    
    # used for training purposes only
    """
        time_track_list.append(episode_time)
        if (i+1) % 3 == 0:
            epsilon -= 0.02
            agent.train_action_nn()
    
    with open('time_data_value.txt', 'w') as f:
        f.write(str(time_track_list))
    
    with open('magnitude_trained.npy', 'wb') as f:
        np.save(f, agent.magnitude_table)
    
    with open('value_trained.npy', 'wb') as f:
        np.save(f, agent.value_table)
    """

    env.close()
