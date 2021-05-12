from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, \
                                                row_col, adjacent_positions, translate, min_distance
from sklearn.model_selection import train_test_split
from agents.short_sight_agent_no_food import ShortSightAgentNoFood
from agents.greedy_agent import GreedyAgent
from utils.helpers import action_to_target
import numpy as np
from copy import deepcopy

steps_per_ep = 200
nb_opponents = 7

env = make("hungry_geese", debug=False)
config = env.configuration

validation_ratio = 0

initial_learning_rate = 0.01
def food_G(rewards):
    rewards_back = rewards[::-1]
    v_prime = 0
    g = []
    for reward in rewards_back:
        v = reward + 0.75*v_prime
        v_prime = v
        g.append(v)
    return g[::-1]

def step_G(rewards):
    rewards_back = rewards[::-1]
    v_prime = 0
    g = []
    for reward in rewards_back:
        v = reward + 0.1*v_prime
        v_prime = v
        g.append(v)
    return g[::-1]

def G(food_rewards, step_rewards):
    return (np.array(food_G(food_rewards)) + step_G(np.array(step_rewards))).reshape(-1, 1)

def transform_sample(samples):
    nb_samples = len(samples)
    if validation_ratio > 0:
        train, test = train_test_split(range(nb_samples), test_size=validation_ratio)
    else:
        train = np.random.choice(nb_samples, nb_samples, replace=False)
        test = np.array([])

    numerical = np.concatenate([sample['cur_state'][1].reshape(1, 4) for sample in samples], axis=0)
    embedding = np.concatenate([sample['cur_state'][0].reshape(1, 24) for sample in samples],
                               axis=0)
    step_reward = [sample['step_reward'] for sample in samples]
    food_reward = [sample['food_reward'] for sample in samples]
    g = np.array(step_reward).reshape(-1, 1)
    g = (g-np.mean(g)) / (np.std(g) + 1E-5)
    y = np.concatenate([sample['action'].reshape(1, 4) for sample in samples], axis=0)
    return [numerical[train], embedding[train], g[train]], y[train],\
           [numerical[test] if len(test) > 0 else np.array([]),
            embedding[test] if len(test) > 0 else np.array([]),
            g[test] if len(test) > 0 else np.array([])],\
           y[test] if len(test) > 0 else np.array([])

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
            food_reward = 2
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

    agent = ShortSightAgentNoFood(learning_rate=initial_learning_rate)
    best_score = 0
    past_scores = 0.0
    i = 1
    for iteration in range(1, 101):
        samples = []
        avg_duration = []
        nb_games = 100 * (1+int(float(iteration)/10))
        nb_games = min(nb_games, 1000)
        print(f'# games to play: {nb_games}')
        for _ in range(nb_games):
            cur_game = run_game(nb_opponents, agent)
            samples += cur_game
            avg_duration.append(len(cur_game))
        avg_nb_steps = np.mean(avg_duration)
        max_nb_steps = np.max(avg_duration)
        print(f'Average game duration (steps): {avg_nb_steps}')
        print(f'Max game duration (steps): {max_nb_steps}')
        X, y, X_val, y_val = transform_sample(samples)

        if avg_nb_steps > best_score:
            best_score = avg_nb_steps
            print('Saving Weights')
            agent.save_weights('ShortSightAgentNoFood')
        elif avg_nb_steps < past_scores*0.98:
            print('Reducing learning rate')
            agent = ShortSightAgentNoFood(learning_rate=initial_learning_rate*(0.01**i))
            agent.load_weights('ShortSightAgentNoFood')
            i += 1
        past_scores = avg_nb_steps
        agent.fit(X, y, X_val, y_val)

        if i > 4:
            break

