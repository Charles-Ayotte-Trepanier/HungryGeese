from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, \
                                                row_col, adjacent_positions, translate, min_distance
from sklearn.model_selection import train_test_split
from agents.short_sight_agent import ShortSightAgent
from agents.greedy_agent import GreedyAgent
from utils.helpers import action_to_target
import numpy as np
from copy import deepcopy

steps_per_ep = 200
nb_opponents = 3

env = make("hungry_geese", debug=False)
config = env.configuration

validation_ratio = 0.1

def food_G(rewards):
    rewards_back = rewards[::-1]
    v_prime = 0
    g = []
    for reward in rewards_back:
        v = reward + 0.8*v_prime
        v_prime = v
        g.append(v)
    return g[::-1]

def G(food_rewards, step_rewards):
    return (np.array(food_G(food_rewards)) + np.array(step_rewards)).reshape(-1, 1)

def transform_sample(samples):
    nb_samples = len(samples)
    if validation_ratio > 0:
        train, test = train_test_split(range(nb_samples), test_size=validation_ratio)
    else:
        train = np.random.choice(nb_samples, nb_samples, replace=False)
        test = np.array([0])

    numerical = np.concatenate([sample['cur_state'][1].reshape(1, 8) for sample in samples], axis=0)
    embedding = np.concatenate([sample['cur_state'][0].reshape(1, 8) for sample in samples], axis=0)
    step_reward = [sample['step_reward'] for sample in samples]
    food_reward = [sample['food_reward'] for sample in samples]
    g = G(food_reward, step_reward)
    g = (g-np.mean(g)) / (np.std(g) + 1E-5)
    y = np.concatenate([sample['action'].reshape(1, 4) for sample in samples], axis=0)
    return [numerical[train], embedding[train], g[train]], y[train],\
           [numerical[test], embedding[test], g[test]], y[test]

def run_game(nb_opponents, my_agent):
    steps = []
    agents = [my_agent] + [GreedyAgent() for _ in range(nb_opponents)]

    state_dict = env.reset(num_agents=nb_opponents + 1)[0]
    observation = state_dict['observation']

    done = False
    my_agent.last_action = state_dict.action
    prev_food_pos = observation.food
    for step in range(1, steps_per_ep):
        actions = []

        for i, agent in enumerate(agents):
            obs = deepcopy(observation)
            obs['index'] = i
            action = agent(obs, config)
            actions.append(action)

        cur_state = agents[0].stateSpace

        state_dict = env.step(actions)[0]
        observation = state_dict['observation']
        my_goose_ind = observation['index']

        my_goose_length = len(observation['geese'][my_goose_ind])

        action = state_dict['action']
        status = state_dict['status']

        if status != "ACTIVE":
            done = True
            next_state = None
        else:
            next_state = agents[0].getStateSpace(observation, action)

        # Check if my goose died
        if my_goose_length == 0:
            done = True
            reward = 0
            food_reward = 0
        else:
            reward = 1
            cur_head_pos = observation['geese'][my_goose_ind][0]
            if cur_head_pos in prev_food_pos:
                food_reward = 2
            else:
                food_reward = 0
        prev_food_pos = observation.food

        steps.append({'cur_state': cur_state,
                      'action': action_to_target(action),
                      'step_reward': reward,
                      'food_reward': food_reward,
                      'next_state': next_state,
                      'done': done})
        if done:
            break

    return steps


if __name__ == "__main__":

    agent = ShortSightAgent()

    for iteration in range(1, 1001):
        samples = []
        avg_duration = []
        for _ in range(100):
            cur_game = run_game(np.random.randint(3)+1, agent)
            samples += cur_game
            avg_duration.append(len(cur_game))
        print(f'Average game duration (steps): {np.mean(avg_duration)}')
        X, y, X_val, y_val = transform_sample(samples)
        agent.fit(X, y, X_val, y_val)
        if (iteration % 10) == 0:
            agent.save_weights('ShortSightAgent')
