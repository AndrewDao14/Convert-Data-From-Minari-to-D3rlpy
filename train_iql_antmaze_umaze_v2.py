import minari
import numpy as np
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import IQLConfig
from d3rlpy.metrics.evaluators import EnvironmentEvaluator
import gym

#Flatten the Observation space of the environment so that it can be used for evaluating
class FlattenAntMazeObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Replace Dict space with a flat Box space
        orig_space = env.observation_space
        low = np.concatenate([
            orig_space["observation"].low,
            orig_space["achieved_goal"].low,
            orig_space["desired_goal"].low
        ])
        high = np.concatenate([
            orig_space["observation"].high,
            orig_space["achieved_goal"].high,
            orig_space["desired_goal"].high
        ])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        return np.concatenate([
            obs["observation"],
            obs["achieved_goal"],
            obs["desired_goal"]
        ])

minari_dataset = minari.load_dataset("D4RL/antmaze/umaze-v1", download = True)
env = minari_dataset.recover_environment()
env = FlattenAntMazeObs(env)

# Show total number of episodes
print("Number of episodes:", len(minari_dataset))

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

    
dataset = MDPDataset(np.concatenate(observations_list, axis = 0),
                      np.concatenate(actions_list, axis = 0),
                      np.concatenate(rewards_list, axis = 0),
                      np.concatenate(terminasls_list, axis = 0),
                     timeouts=np.concatenate(timeouts_list, axis = 0))

#Create model
iql = IQLConfig(
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    gamma=0.99,
    expectile=0.9,
    weight_temp=10.0, #Beta
    tau=0.005,
    batch_size=256
    
)
#create model
model = iql.create()   

print(print(env.observation_space))
#Evaluator
env_evaluator = EnvironmentEvaluator(env=env, n_trials=10)

#Train model
model.fit(
    dataset=dataset,
    n_steps=1000000,
    n_steps_per_epoch=1000,        # adjust as needed
    save_interval=10,
    experiment_name="IQL_test_2",
    show_progress=True,
    evaluators={"environment": env_evaluator}
)


model.save_model("IQL_test_2.d3")
