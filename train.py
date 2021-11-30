'''
    Uses the Semi-Gradient Episodic SARSA Reinforcement Learning algorithm to train an
    agent to complete OpenAI Gym's implementation of the MountainCar control task.
'''

from helper_file.tile_coding import *
import gym
import numpy as np
import random

# Initialize Simulator
env = gym.make('MountainCar-v0')
num_actions = env.action_space.n
num_observations = env.observation_space.shape[0]

print("Number of Possible Actions:", num_actions)
print("Number of Observations:", num_observations)
print("Max Observation:", env.observation_space.high)
print("Min Observation :", env.observation_space.low)

# Normalize Observations
observation_min = np.array(env.observation_space.low)
observation_norm_max = np.array(env.observation_space.high) - observation_min

def norm_observation(observation):
  return(observation - observation_min)/observation_norm_max


def get_observation(normalized_observation):
  return(normalized_observation * observation_norm_max) + observation_min

# Unit Testing
def test_normalize_function():
    obsv_to_norm_obsv = [([-1.3, 0.08], [-0.05555553, 1.07142857]),
                            ([-1.2, -0.07], [0, 0]),
                            ([0.6, 0.07], [1., 1.])]

    for observation, expected_normalized_observation in obsv_to_norm_obsv:
        np.testing.assert_array_almost_equal(norm_observation(observation),
        expected_normalized_observation)

    for observation, expected_normalized_observation in obsv_to_norm_obsv:
        np.testing.assert_array_almost_equal(expected_normalized_observation,
        norm_observation(observation))

test_normalize_function()

# Setup Tile Coding
num_states = 4096
num_tiles = 4
num_tilings = 32
iht = IHT(num_states)

# Training Parameters
weights = np.zeros((num_actions, num_states))
number_of_episodes = 1000
epsilon = 0.1
step_size = 0.5/num_tilings
gamma = 1

def get_state(observation):
  '''
    Convert continuous observation into aggregate state.
  '''
  observation = num_tiles * norm_observation(observation)
  aggregated_state = tiles(iht, num_tilings, observation)
  curr_state = np.zeros(num_states)
  curr_state[aggregated_state] = 1
  return curr_state


def action_value(state, action):
  '''
    Get action value for state action pair.
  '''
  multiple = weights[action] * state
  return np.sum(multiple)


def epsilon_greedy_action_selection(observation):
  '''
    Returns action with greates action_value
    Select random actions with a probability of 1 - epsilon.
  '''
  state = get_state(observation)

  if random.random() >= epsilon:
    max_value =  float("-inf")
    max_action = 0

    for action in range(num_actions):
      temp_value = action_value(state, action)
      if temp_value > max_value:
        max_action = action
        max_value = temp_value

    return max_action, state

  else:
    action = random.choice(range(num_actions))
    return action, state


def begin_episode():
    '''
        Agent's first step in an episode.
    '''
    observation = env.reset()
    action, state = epsilon_greedy_action_selection(observation)
    return action, state


def end_episode(state, action, reward):
    '''
        Performs update for agent's last step in an episode.
    '''
    weights[action] += step_size * (reward - action_value(state, action)) * state


def take_step(next_observation, state, action, reward):
    '''
        Performs update for each step an agent takes.
    '''
    next_action, next_state = epsilon_greedy_action_selection(next_observation)
    weights[action] += step_size * (reward + (gamma * action_value(next_state, next_action)) - action_value(state, action)) * state

    return next_action, next_state
 

# Start Training
def train_agent():
    '''
        Starts training agent.
    '''
    for _ in range(number_of_episodes):
        action, state = begin_episode()

        while True:
            # Perform action in simulator
            next_observation, reward, end_of_episode, _ = env.step(action)
            env.render()

            if not end_of_episode:
                action, state = take_step(next_observation, state, action, reward)
            else:
                end_episode(state, action, reward)
                break
train_agent()