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
    agent.initialize_value_table()
    column_index = {"move ": 0, "turn ": 1, "attack ": 2, "pitch ": 3, "jump ": 4}

    for i in range(args.episodes):
        print("reset " + str(i))
        obs = env.reset()

        steps = 0
        done = False
        rewards = Reward()
        state_space = [0, 6, 7, 0, 1]
        previous_info = 0
        while not done and (args.episodemaxsteps <= 0 or steps < args.episodemaxsteps):

            try:
                previous_info_loaded = json.loads(previous_info)
                if previous_info_loaded.get("LineOfSight").get("hitType") == "entity" and previous_info_loaded.get("LineOfSight").get("inRange") == True:
                    action_type = "attack "
                else: 
                    action_type = agent.action_selection_nn(state_space)
            except:
                action_type = agent.action_selection_nn(state_space)
            magnitude = agent.magnitude_selection_nn(state_space, action_type)

            action = action_type + str(magnitude)

            print(f"action that is taken: {action}")
            obs, reward, done, info = env.step(action)
            steps += 1
            custom_reward = rewards.calculate_reward(info)
            previous_info = info
        
            try:
                loaded_info = json.loads(info)
                if loaded_info.get("MobsKilled") == 1:
                    done = True
            except:
                pass
            
            print("reward: " + str(custom_reward))
            print("done: " + str(done))
            print(type(info))
            print(info)
            state_space = agent.state_space_function(info)
            print(f"current state space: {state_space}")
            state_space_index = agent.state_space_index(state_space)
            print(f"current state space index: {state_space_index}")
            print(f"current value function for state space: {agent.value_table[state_space_index]}")
            if action_type in ["move ", "turn ", "attack "]:
                env.step(action_type + "0")
            

            if custom_reward > agent.value_table[state_space_index][column_index.get(action_type)]:
                agent.value_table[state_space_index][column_index.get(action_type)] = custom_reward
                if custom_reward >= max(agent.value_table[state_space_index]):
                    agent.train_action_nn()
                try:
                    agent.magnitude_table[state_space_index][column_index.get(action_type)] = int(magnitude)
                except:
                    agent.magnitude_table[state_space_index][column_index.get(action_type)] = float(magnitude)
                # retrain magitude_selection_nn

            if args.saveimagesteps > 0 and steps % args.saveimagesteps == 0:
                h, w, d = env.observation_space.shape
                img = Image.fromarray(obs.reshape(h, w, d))
                img.save('image' + str(args.role) + '_' + str(steps) + '.png')

            time.sleep(0.05)


    env.close()
