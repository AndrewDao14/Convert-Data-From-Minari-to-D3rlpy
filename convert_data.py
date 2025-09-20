import minari
import numpy as np


#Download dataset
minari_dataset = minari.load_dataset("D4RL/antmaze/umaze-v1", download = True)


#Process Minari data
observations_list = []
actions_list = []
rewards_list = []
terminasls_list = []
timeouts_list = []

for episode in minari_dataset.iterate_episodes():

    obs = episode.observations["observation"]
    achieved = episode.observations["achieved_goal"]
    desired = episode.observations["desired_goal"]
    action = episode.actions
    reward = episode.rewards
    termination = episode.terminations
    trunctation = episode.truncations

    #Add to list
    obs_achieved = np.concatenate([obs, achieved, desired], axis=1)
    observations_list.append(np.array(obs_achieved[:-1]))
    actions_list.append(action)
    rewards_list.append(reward)
    terminasls_list.append(termination)
    timeouts_list.append(trunctation)

#Now we have five list with each element containing one full episode
for i in range(5):  # show first 5 rows
    print(f"Step {i}:")
    print("  Observation:", observations_list[0][i])
    print("  Action:", actions_list[0][i])
    print("  Reward:", rewards_list[0][i])
    print("  Terminal:", terminasls_list[0][i])
    print("  Timeout:", timeouts_list[0][i])
    print()


## OPTIONAL: CONVERT INTO AN MDPDataset TO BE USED BY d3rlpy
from d3rlpy.dataset import MDPDataset
dataset = MDPDataset(np.concatenate(observations_list, axis = 0),
                      np.concatenate(actions_list, axis = 0),
                      np.concatenate(rewards_list, axis = 0),
                      np.concatenate(terminasls_list, axis = 0),
                     timeouts=np.concatenate(timeouts_list, axis = 0))