from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, \
                                                row_col, adjacent_positions, translate, min_distance
from agents.last_action_agent import LastActionAgent
from agents.greedy_agent import GreedyAgent
from utils.helpers import action_to_target
import numpy as np
from copy import deepcopy

steps_per_ep = 200
nb_opponents = 3

env = make("hungry_geese", debug=False)
config = env.configuration

def transform_sample(samples):
    nb_samples = len(samples)
    shuffled = np.random.choice(nb_samples, int(0.9*nb_samples), replace=False)

    state = np.concatenate([sample['cur_state'].reshape(1, 4) for sample in samples], axis=0)
    g = np.array([np.array(sample['reward'], dtype=float) for sample in samples]).reshape(-1, 1)
    g = (g-np.mean(g)) / (np.std(g) + 1E-5)
    y = np.concatenate([sample['action'].reshape(1, 4) for sample in samples], axis=0)
    return [state[shuffled], g[shuffled]], y[shuffled],\
           [state[~shuffled], g[~shuffled]], y[~shuffled]

def run_game(nb_opponents, my_agent):
    steps = []
    agents = [my_agent] + [GreedyAgent() for _ in range(nb_opponents)]

    state_dict = env.reset(num_agents=nb_opponents + 1)[0]
    observation = state_dict['observation']

    done = False
    my_agent.last_action = state_dict.action
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

        steps.append({'cur_state': cur_state,
                      'action': action_to_target(action),
                      'reward': reward,
                      'next_state': next_state,
                      'done': done})
        if done:
            break

    return steps


if __name__ == "__main__":

    agent = LastActionAgent()

    for _ in range(100):
        samples = []
        avg_duration = []
        for _ in range(1000):
            cur_game = run_game(1, agent)
            samples += cur_game
            avg_duration.append(len(cur_game))
        print(f'Average game duration (steps): {np.mean(avg_duration)}')
        X, y, X_val, y_val = transform_sample(samples)
        agent.fit(X, y, X_val, y_val)

    agent.save_weights('LastActionAgent')
