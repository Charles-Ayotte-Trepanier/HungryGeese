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
batch_size = int(1E9)
epoch = 10
initial_learning_rate = 0.1

def compute_G(rewards, discount):
    rewards_back = rewards[::-1]
    v_prime = 0
    g = []
    for reward in rewards_back:
        v = reward + discount*v_prime
        v_prime = v
        g.append(v)
    return g[::-1]

def transform_sample(samples):
    nb_samples = len(samples)
    if validation_ratio > 0:
        train, test = train_test_split(range(nb_samples), test_size=validation_ratio)
    else:
        train = np.random.choice(nb_samples, nb_samples, replace=False)
        test = np.array([])

    forbidden = np.concatenate([sample['cur_state'][1].reshape(1, 4) for sample in samples], axis=0)
    top = np.concatenate([sample['cur_state'][0][0].reshape(1, 21) for sample in samples], axis=0)
    right = np.concatenate([sample['cur_state'][0][1].reshape(1, 21) for sample in samples], axis=0)
    bottom = np.concatenate([sample['cur_state'][0][2].reshape(1, 21) for sample in samples],
                            axis=0)
    left = np.concatenate([sample['cur_state'][0][3].reshape(1, 21) for sample in samples], axis=0)

    step_reward = [sample['step_reward'] for sample in samples]
    food_reward = [sample['food_reward'] for sample in samples]
    step_G = compute_G(step_reward, 0.8)
    food_G = compute_G(food_reward, 0.9)
    g = np.array(step_G).reshape(-1, 1) + np.array(food_G).reshape(-1, 1)
    g = (g-np.mean(g)) / (np.std(g) + 1E-5)
    y = np.concatenate([sample['action'].reshape(1, 4) for sample in samples], axis=0)
    return [forbidden[train], top[train], right[train], bottom[train], left[train],
            g[train]], y[train],\
           [forbidden[test] if len(test) > 0 else np.array([]),
            top[test] if len(test) > 0 else np.array([]),
            right[test] if len(test) > 0 else np.array([]),
            bottom[test] if len(test) > 0 else np.array([]),
            left[test] if len(test) > 0 else np.array([]),
            g[test] if len(test) > 0 else np.array([])],\
           y[test] if len(test) > 0 else np.array([])

def run_game(nb_opponents, my_agent):
    steps = []
    agents = [my_agent] + [GreedyAgent() for _ in range(nb_opponents)]

    state_dict = env.reset(num_agents=nb_opponents + 1)[0]
    observation = state_dict['observation']

    done = False
    my_agent.last_action = state_dict.action

    prev_len = 1
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

        # negative reward for crashing into another goose
        if (my_goose_length == 0):
            done = True
            if (prev_len == 1) and ((observation.step % 40) == 0):
                no_crash_reward = 0
            else:
                no_crash_reward = -3
        else:
            no_crash_reward = 0

        if (my_goose_length > prev_len) or \
                ((my_goose_length == prev_len) and (observation.step % 40) == 0):
            food_reward = 2
        else:
            food_reward = 0
        prev_len = my_goose_length

        steps.append({'cur_state': cur_state,
                      'action': action_to_target(action),
                      'step_reward': no_crash_reward,
                      'food_reward': food_reward,
                      'next_state': next_state,
                      'done': done})
        if done:
            break

    return steps, 1 if my_goose_length > 0 else 0


if __name__ == "__main__":

    agent = ShortSightAgentNoFood(learning_rate=initial_learning_rate)
    # embeddings = np.array([1, -1]).reshape(2, 1)
    # dense = np.array([4, 10, 4]).reshape(3, 1)
    # bias = np.array([0]).reshape(1, )
    # agent.model.set_weights([embeddings, dense, bias])
    #agent.load_weights('ShortSightAgentNoFood')
    best_score = 0
    best_wins = 0
    i = 1
    come_back = 0
    for iteration in range(1, 101):
        samples = []
        avg_duration = []
        wins = []
        nb_games = 100 * (1+int(float(iteration)/10))
        nb_games = min(nb_games, 1000)
        print(f'# games to play: {nb_games}')
        for _ in range(nb_games):
            cur_game, win = run_game(nb_opponents, agent)
            samples += cur_game
            wins.append(win)
            avg_duration.append(len(cur_game))
        avg_nb_steps = np.mean(avg_duration)
        max_nb_steps = np.max(avg_duration)
        avg_win = np.mean(wins)
        print(f'Average game duration (steps): {avg_nb_steps}')
        print(f'Max game duration (steps): {max_nb_steps}')
        print(f'Win rate: {avg_win}')
        X, y, X_val, y_val = transform_sample(samples)

        if (avg_nb_steps > best_score) or (avg_win > best_wins):
            if avg_nb_steps > best_score:
                best_score = avg_nb_steps
            if avg_win > best_wins:
                best_wins = avg_win
            print('Saving Weights')
            agent.save_weights('ShortSightAgentNoFood')
            X_best = X
            y_best = y
            X_val_best = X_val
            y_val_best = y_val
            agent.fit(X, y, X_val, y_val, batch_size=batch_size, epoch=epoch)
        elif come_back > 0:
            agent.fit(X, y, X_val, y_val, batch_size=batch_size, epoch=epoch)
            come_back -= 1
        elif (avg_nb_steps < best_score*0.95) and (avg_win < best_wins*0.95):
            print('Reducing learning rate')
            agent = ShortSightAgentNoFood(learning_rate=initial_learning_rate*(0.1**i))
            agent.load_weights('ShortSightAgentNoFood')
            agent.fit(X_best, y_best, X_val_best, y_val_best, batch_size=batch_size, epoch=epoch)
            i += 1
            come_back = 5
        else:
            agent.fit(X, y, X_val, y_val, batch_size=batch_size, epoch=epoch)

        if i > 5:
            break

