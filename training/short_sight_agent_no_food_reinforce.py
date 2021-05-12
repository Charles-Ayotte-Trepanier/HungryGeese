from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, \
                                                row_col, adjacent_positions, translate, min_distance
from agents.short_sight_agent_no_food import ShortSightAgentNoFood
from agents.greedy_agent import GreedyAgent
from utils.helpers import action_to_target
import numpy as np
from copy import deepcopy

steps_per_ep = 200
nb_opponents = 3

env = make("hungry_geese", debug=True)
config = env.configuration

def food_G(rewards):
    rewards_back = rewards[::-1]
    v_prime = 0
    g = []
    for reward in rewards_back:
        v = reward + 0.75*v_prime
        v_prime = v
        g.append(v)
    return g[::-1]

def G(food_rewards, step_rewards):
    return (np.array(food_G(food_rewards)) + np.array(step_rewards)).reshape(-1, 1)

def transform_sample(samples):
    nb_samples = len(samples)
    shuffled = np.random.choice(nb_samples, int(0.9*nb_samples), replace=False)

    numerical = np.concatenate([sample['cur_state'][1].reshape(1, 4) for sample in samples], axis=0)
    embedding = np.concatenate([sample['cur_state'][0].reshape(1, 8) for sample in samples], axis=0)
    step_reward = [sample['step_reward'] for sample in samples]
    food_reward = [sample['food_reward'] for sample in samples]
    g = np.array(step_reward).reshape(-1, 1)
    g = (g-np.mean(g)) / (np.std(g) + 1E-5)
    y = np.concatenate([sample['action'].reshape(1, 4) for sample in samples], axis=0)
    return [numerical[shuffled], embedding[shuffled], g[shuffled]], y[shuffled],\
           [numerical[~shuffled], embedding[~shuffled], g[~shuffled]], y[~shuffled]

def run_game(nb_opponents, my_agent):
    steps = []
    agents = [my_agent] + [GreedyAgent() for _ in range(nb_opponents)]

    state_dict = env.reset(num_agents=nb_opponents + 1)[0]
    observation = state_dict['observation']

    done = False
    my_agent.last_action = state_dict.action
    prev_food_eaten = 0
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
        else:
            reward = 1

        cur_food_eaten = state_dict.reward % 100
        if cur_food_eaten > prev_food_eaten:
            food_reward = 5
        else:
            food_reward = 0
        prev_food_eaten = cur_food_eaten

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

    agent = ShortSightAgentNoFood()

    for _ in range(100):
        samples = []
        avg_duration = []
        for _ in range(1000):
            cur_game = run_game(np.random.randint(3)+1, agent)
            samples += cur_game
            avg_duration.append(len(cur_game))
        avg_steps_per_game = {np.mean(avg_duration)}
        print(f'Average game duration (steps): {avg_steps_per_game}')
        if avg_steps_per_game >= 30:
            break
        X, y, X_val, y_val = transform_sample(samples)
        agent.fit(X, y, X_val, y_val)

    agent.save_weights('ShortSightAgent_no_food')
