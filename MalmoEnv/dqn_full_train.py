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
import csv
import os
from json import JSONDecodeError

import malmoenv
import argparse
from pathlib import Path
import time

import numpy as np
from PIL import Image
import json
from xml.etree import ElementTree

from agents.dqn_agent import DQNAgent
from reinforcement import State, Reward
import utils

COMMANDS_LIST = ['move 1', 'pitch 1', 'pitch -1', 'turn 1', 'turn -1', 'attack 1',
                  'attack 0', 'turn 0', 'move 0']
COMMANDS_DICT = {index: obj for index, obj in enumerate(COMMANDS_LIST)}

STATE_SIZE = 4
ACTION_SIZE = 6
MS_PER_TICK = 10
EPISODE_RESULTS_FILE = "dqn_full_v4_episode_results.csv"
AGENT_SAVE_FILE = "dqn_full_v4_agent.pkl"
IS_TRAINING = True
DO_SAVE = False

def save_episode_results(epsilon, episode_length, cumulative_reward):
    if IS_TRAINING and DO_SAVE:
        with open(EPISODE_RESULTS_FILE, 'a', newline='') as file:
            csv_writer = csv.writer(file)

            # If the file is empty, write the header
            if file.tell() == 0:
                csv_writer.writerow(['epsilon', 'episode_length', 'cumulative_reward'])

            # Append the data
            csv_writer.writerow([epsilon, episode_length, cumulative_reward])

def log(msg):
    with open('log.txt', 'a') as file:
        file.write(f'{msg}\n')

def save_agent(agent):
    if IS_TRAINING and DO_SAVE:
        agent.save(AGENT_SAVE_FILE)

def load_agent():
    agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
    if IS_TRAINING and os.path.exists(AGENT_SAVE_FILE):
        print("Previous agent exists, loading it...")
        agent.load(AGENT_SAVE_FILE)
    else:
        print("Previous agent doesn't exist, creating new...")

    return agent

def set_tick_mission_xml():
    tree = ElementTree.parse("missions/chicken_deletion_5x5.xml")
    root = tree.getroot()

    mod_settings = root.find(".//{http://ProjectMalmo.microsoft.com}ModSettings")
    ms_per_tick_element = mod_settings.find(".//{http://ProjectMalmo.microsoft.com}MsPerTick")
    ms_per_tick_element.text = str(MS_PER_TICK)

    ElementTree.register_namespace("", "http://ProjectMalmo.microsoft.com")

    tree.write("missions/chicken_deletion_5x5.xml")
    print(f"Set mission xml MsPerTick to {MS_PER_TICK}")

def wait(frames=1):
    time.sleep((MS_PER_TICK / 1000) * frames)

def spawn_chicken(env, start, end, step=0.5):
    interval_values = np.arange(start, end + step, step)
    xz = np.random.choice(interval_values, 2, replace=False)
    print(f"Spawning Chicken at {xz[0]} 4 {xz[1]}")
    env.step(f"chat /summon Chicken {xz[0]} 4 {xz[1]}")
    wait()

def no_op_env_init(env):
    time.sleep(1)
    spawn_chicken(env, -6.5, 6.5)

    while True:
        try:
            rlState = None
            data = None

            obs, reward, done, info = env.step('move 0')
            wait()

            data = json.loads(info)
            rlState = State(data)
            if rlState == None:
                print("State was None. Retrying...")
                continue
            else:
                return rlState, data

        except:
            print("Error initializing environment. Retrying...")
            continue


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='malmovnv test')
    parser.add_argument('--mission', type=str, default='missions/chicken_deletion.xml', help='the mission xml')
    parser.add_argument('--port', type=int, default=9000, help='the mission server port')
    parser.add_argument('--server', type=str, default='127.0.0.1', help='the mission server DNS or IP address')
    parser.add_argument('--port2', type=int, default=None,
                        help="(Multi-agent) role N's mission port. Defaults to server port.")
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

    set_tick_mission_xml()
    xml = Path(args.mission).read_text()
    env = malmoenv.make()

    env.init(xml, args.port,
             server=args.server,
             server2=args.server2, port2=args.port2,
             role=args.role,
             exp_uid=args.experimentUniqueId,
             episode=args.episode, resync=args.resync,
             action_filter=['move', 'pitch', 'turn', 'attack'])

    agent = load_agent()
    # agent.epsilon = 1.0
    # from collections import deque
    # agent.memory.memory = deque([], 10000)
    # agent.epsilon_min = 0.15
    # save_agent(agent)
    # quit()

    try:
        for i in range(args.episodes):
            print("reset " + str(i))
            obs = env.reset()
            rlState, data = no_op_env_init(env)
            state = rlState.to_state()

            total_reward = 0
            current_damage_dealt = 0
            previous_coordinates = round(data['XPos'], 3), round(data['ZPos'], 3)
            previous_yaw = rlState.get_yaw()
            previous_pitch = rlState.get_pitch()
            steps = 0
            done = False
            while not done and (args.episodemaxsteps <= 0 or steps < args.episodemaxsteps):
                print(f">> TIMESTEP {steps}")
                action = agent.act(state)
                print("action:", COMMANDS_DICT[action])
                env.step(COMMANDS_DICT[action])
                wait(1)

                obs, reward, done, info = None, None, None, None
                if action == "move 1" or COMMANDS_DICT.get(action, False) == "move 1":
                    wait(2)
                    obs, reward, done, info = env.step("move 0")
                    wait(9)
                    obs, reward, done, info = env.step("move 0")
                elif action == "attack 1" or COMMANDS_DICT.get(action, False) == "attack 1":
                    wait(2)
                    obs, reward, done, info = env.step("attack 0")
                    wait(5)
                    obs, reward, done, info = env.step("attack 0")
                elif (action == "pitch 1" or COMMANDS_DICT.get(action, False) == "pitch 1") or \
                     (action == "pitch -1" or COMMANDS_DICT.get(action, False) == "pitch -1"):
                    wait(2)
                    obs, reward, done, info = env.step("pitch 0")
                    wait(2)
                    obs, reward, done, info = env.step("pitch 0")
                elif (action == "turn 1" or COMMANDS_DICT.get(action, False) == "turn 1") or \
                        (action == "turn -1" or COMMANDS_DICT.get(action, False) == "turn -1"):
                    wait(2)
                    obs, reward, done, info = env.step("turn 0")
                    wait(2)
                    obs, reward, done, info = env.step("turn 0")
                steps += 1

                # Convert the JSON string to a Python object (dictionary in this case)
                try:
                    data = json.loads(info)
                except (JSONDecodeError, TypeError) as e:
                    print("JSON ERROR")
                    continue

                # Retrieve the current state
                try:
                    rlState = State(data)
                except KeyError as e:
                    print("KEY ERROR")
                    continue

                next_state = rlState.to_state()
                distance_from_chicken = utils.distance(rlState.get_xz_delta())
                current_coordinates = rlState.get_coordinates()
                print("xz_delta:", rlState.get_xz_delta())
                print("changeInCoords:", (rlState.get_coordinates()[0] - previous_coordinates[0], rlState.get_coordinates()[1] - rlState.get_coordinates()[1]))
                print("changeinYaw:", rlState.get_yaw() - previous_yaw)
                print("changeInPitch:", rlState.get_pitch() - previous_pitch)
                print("current coords:", current_coordinates)

                print("state:", next_state)
                print("distance:", distance_from_chicken)

                # Calculate reward
                rlReward = Reward(data, rlState, action, previous_coordinates, current_damage_dealt)
                reward, killedChicken, inSync, penalty_step = rlReward.calculate_reward()
                if killedChicken:
                    done = True
                    if not inSync:
                        print(f"KILLED CHICKEN FROM STATE {state} WITH ACTION: {COMMANDS_DICT.get(action)}")
                        log(f"KILLED CHICKEN FROM STATE {state} WITH ACTION: {COMMANDS_DICT.get(action)}")
                        continue
                if penalty_step is not None:
                    env.step(penalty_step)
                    wait()

                print(f">> WORLDTIME: {data['TotalTime']}")
                print(f">> TIMESTEP {steps} REWARD: " + str(reward))

                agent.remember(state, action, reward, next_state, done)
                total_reward += reward
                previous_coordinates = current_coordinates
                previous_yaw = rlState.get_yaw()
                previous_pitch = rlState.get_pitch()
                current_damage_dealt = data['DamageDealt']

                state = next_state
                agent.replay(steps)

                if args.saveimagesteps > 0 and steps % args.saveimagesteps == 0:
                    h, w, d = env.observation_space.shape
                    img = Image.fromarray(obs.reshape(h, w, d))
                    img.save('image' + str(args.role) + '_' + str(steps) + '.png')

                print()

            print(f"Epsilon value: {agent.epsilon}")
            print("Episode {}: Total Reward: {}".format(i, total_reward))
            agent.decay_epsilon()

            save_agent(agent)
            save_episode_results(agent.epsilon, steps, total_reward)

        # Save the DQN state and other necessary information
        save_agent(agent)
        env.close()
    except Exception as e:
        print(f"exception caught while training. restarting mission...")
        raise e
