# NOTE: All of our relevant code is in this folder, ChickenDeletion/MalmoEnv: However, in order to run the project properly, you must clone the project from the root folder, https://github.com/briannlin/ChickenDeletion/tree/master.

# Project Files Overview #

### Jupyter Notebook Files
- project.ipynb 
  - Demonstrates the Q-table agent and the DDQN agent performing their optimal states in response to a certain state, and evaluation graphs containing performance results over training.
- project.html
  - HTML file of the project.ipynb file containing code and cell outputs.

### Malmo Mission Files
- missions/chicken_deletion.xml
  - Contains the environmental setup for training the agent, which is a 15x15 enclosed area with the agent and chickens spawning within, as well as other Malmo configurations.
- missions/chicken_deletion_5x5.xml
  - Contains the environmental setup for training the agent, which is a 15x15 enclosed area with the agent and chickens spawning within, as well as other Malmo configurations.

### Q-Table Agent Files
- qtable_helper_functions.py
  - Contains the agent class and the reward class to help with computations related to state space, reward functions, and decision making.
- value_trained.npy	
  - Contains the q-table after training with state space for rows, action as columns, and state-action reward within the cells.
- time_data_value.npy
  - Contains the list of time needed to kill the chicken during training of best action policy.
- time_data_magnitude.npy
  - Contains the list of time needed to kill the chicken during training of optimal magnitude given state-action.
- qtable_run.py
  - Main file for training the q-table agent, which utilizes epsilon exploration.
- magnitude_trained.npy
  - Contains the q-table after training with state space for rows, action as columns, and magnitude for the action given state space within the cells.

### DDQN Agent Files
- agents/dqn_agent.py
  - Contains the DDQN algorithm and agent, with features such as epsilon-exploration, choosing an optimal action, experience replay buffer, gradient descent to minimize loss and update the policy and target network, and save/load functionality.
- dqn_full_train.py
  - Contains the environment training loop used to train the DDQN agent to perform optimal moves according to the DDQN algorithm by using states and reward shaping.
- dqn_full_v4_agent.pkl
  - The fully-trained DDQN agent is saved in this .pkl file and is loaded in the .ipynb demonstration.
- dqn_full_v4_episode_results.csv
  - The results of the DDQN agent during each episode of training containing epsilon value, cumulative reward, and episode length - these results are used in the .ipynb demo evaluation.
- dqn_qtable_baseline_full_train.py
  - Contains the environment training loop used to train the baseline Q-table agent (used for comparison against the DDQN agent) to perform optimal moves by using states and reward shaping.
- dqn_qtable_baseline_full_simplified_agent.pkl
  - Contains the environment training loop used to train the baseline Q-table agent (used for comparison against the DDQN agent) to perform optimal moves according to the basic Q-table algorithm by using states and reward shaping.
- dqn_qtable_baseline_full_simplified_episode_results.csv
  - The results of the baseline Q-table agent (used for comparison against the DDQN agent) during each episode of training containing epsilon value, cumulative reward, and episode length - these results are used in the .ipynb demo evaluation.
- reinforcement.py
  - For the DDQN agent’s reinforcement learning - contains a custom State and Reward class used to generate a state representation and a total episode reward given environment observation data.
- utils.py
  - Contains utility helper functions for the DDQN agent such as custom rounding, calculating distance and angles, etc.


# MalmoEnv Setup + Run Instructions #

MalmoEnv is an OpenAI "gym" Python Environment for Malmo/Minecraft, directly implemented Python to Java Minecraft.

A Python "gym env" can be created and used to run an agent in a Malmo mission. Each such env has a remote Minecraft instance
associated to it (by DNS name or IP and Port). For multi-agent missions, the first agent's (role 0) Minecraft 
client instance is used as a coordinator to allow all agents to rendezvous on mission starts (i.e. on env resets).

As it's pure Python, you just need this one package, its direct dependencies and (Java) Minecraft! Example missions, including some from the 2018 MarLo competition can be found in the "missions" directory.

## Examples of use: ##

Install dependencies:

Java8 JDK ([AdoptOpenJDK](https://adoptopenjdk.net/)), python3, git

`pip3 install gym lxml numpy pillow`

To prepare Minecraft (after cloning this repository with 
`git clone https://github.com/Microsoft/malmo.git`):

`cd malmo/Minecraft`

`(echo -n "malmomod.version=" && cat ../VERSION) > ./src/main/resources/version.properties` 

Running a single agent example mission (run each command in different cmd prompt/shells - use launchClient.bat on Windows):

`./launchClient.sh -port 9000 -env` or (On Windows) `launchClient.bat -port 9000 -env`

(In another shell) `cd malmo/MalmoEnv` optionally run `python3 setup.py install`

`python3 <run.py/dqn_full_train.py> --mission missions/chicken_deletion.xml --port 9000 --episodes 10`
