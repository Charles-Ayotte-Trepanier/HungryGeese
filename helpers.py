from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, \
                                                                row_col, adjacent_positions, translate, min_distance


import numpy as np
import pickle
from random import choice

from copy import deepcopy


def geese_heads(obs_dict, config_dict):
    """
    Return the position of the geese's heads
    """
    configuration = Configuration(config_dict)

    observation = Observation(obs_dict)
    player_index = observation.index
    player_goose = observation.geese[player_index]
    player_head = player_goose[0]
    player_row, player_column = row_col(player_head, configuration.columns)
    positions = []
    for geese in observation.geese:
        if len(geese) > 0:
            geese_head = geese[0]
            row, column = row_col(geese_head, configuration.columns)
        else:
            row = None
            column = None
        positions.append((row, column))
    return positions


def get_last_actions(previous_geese_heads, heads_positions):
    def get_last_action(prev, cur):
        last_action = None

        prev_row = prev[0]
        prev_col = prev[1]
        cur_row = cur[0]
        cur_col = cur[1]

        if cur_row is not None:
            if (cur_row - prev_row == 1) | ((cur_row == 0) & (prev_row == 6)):
                last_action = Action.SOUTH.name
            elif (cur_row - prev_row == -1) | ((cur_row == 6) & (prev_row == 0)):
                last_action = Action.NORTH.name
            elif (cur_col - prev_col == 1) | ((cur_col == 0) & (prev_col == 10)):
                last_action = Action.EAST.name
            elif (cur_col - prev_col == -1) | ((cur_col == 10) & (prev_col == 0)):
                last_action = Action.WEST.name

        return last_action

    if len(previous_geese_heads) == 0:
        actions = [Action.SOUTH.name, Action.NORTH.name, Action.EAST.name, Action.WEST.name]
        nb_geeses = len(heads_positions)
        last_actions = ["None" for _ in range(nb_geeses)]
    else:
        last_actions = [get_last_action(*pos) for pos in zip(previous_geese_heads, heads_positions)]

    return last_actions


def central_state_space(obs_dict, config_dict, last_actions):
    """
    Recreating a board where my agent's head in the middle of the board
    (position (4,5)), and creating features accordingly
    """

    last_actions_dict = {
        Action.WEST.name: 1,
        Action.EAST.name: 2,
        Action.NORTH.name: 3,
        Action.SOUTH.name: 4,
        "None": 16
    }

    configuration = Configuration(config_dict)

    observation = Observation(obs_dict)
    player_index = observation.index
    player_goose = observation.geese[player_index]
    player_head = player_goose[0]
    longuest_opponent = 0
    for i, goose in enumerate(observation.geese):
        if i != player_index:
            opponent_length = len(goose)
            if opponent_length > longuest_opponent:
                longuest_opponent = opponent_length
    player_row, player_column = row_col(player_head, configuration.columns)
    row_offset = player_row - 3
    column_offset = player_row - 5

    foods = observation['food']

    def centralize(row, col):
        if col > player_column:
            new_col = (5 + col - player_column) % 11
        else:
            new_col = 5 - (player_column - col)
            if new_col < 0:
                new_col += 11

        if row > player_row:
            new_row = (3 + row - player_row) % 7
        else:
            new_row = 3 - (player_row - row)
            if new_row < 0:
                new_row += 7
        return new_row, new_col

    food1_row, food1_column = centralize(*row_col(foods[0], configuration.columns))
    food2_row, food2_column = centralize(*row_col(foods[1], configuration.columns))

    food1_row_feat = float(food1_row - 3) / 5 if food1_row >= 3 else float(food1_row - 3) / 5
    food2_row_feat = float(food2_row - 3) / 5 if food2_row >= 3 else float(food2_row - 3) / 5

    food1_col_feat = float(food1_column - 5) / 5 if food1_column >= 5 else float(
        food1_column - 5) / 5
    food2_col_feat = float(food2_column - 5) / 5 if food2_column >= 5 else float(
        food2_column - 5) / 5

    # Create the grid
    board = np.zeros([7, 11])
    # Add food to board
    board[food1_row, food1_column] = 15
    board[food2_row, food2_column] = 15

    for geese_id, geese in enumerate(observation.geese):
        nb_blocks = len(geese)
        if nb_blocks > 0:
            for i, pix in enumerate(geese[::-1]):
                if ((i + 1) == nb_blocks):  # This is the head
                    idx = last_actions_dict[last_actions[geese_id]]  # head
                else:
                    idx = (i + 5) if (i + 5) < 15 else 14
                row, col = centralize(*row_col(pix, configuration.columns))
                board[row, col] = idx

    return board, len(
        player_goose), longuest_opponent, food1_row_feat, food1_col_feat, food2_row_feat, food2_col_feat



actions_list = np.array(['EAST',
                        'WEST',
                        'SOUTH',
                        'NORTH'])
def action_to_target(action):
    pos = np.argmax(actions_list == action)
    target = np.zeros(4)
    target[pos] = 1
    return target

def target_to_action(target):
    pos = np.argmax(target)
    return actions_list[pos]

def pred_to_action(pred):
    pos = np.argmax(np.random.multinomial(1, pred))
    return actions_list[pos]

def pred_to_action_greedy(pred):
    pos = np.argmax(pred)
    return actions_list[pos]



def add_numerical(steps):
    numerical = ['goose_size',
                 'longuest_opponent',
                 'hunger',
                 'step']
    food_position = ['food1_col',
                     'food2_col',
                     'food1_row',
                     'food2_row']
    for step in steps:
        numerical_vector = np.zeros(len(numerical)+1)
        for i, nm in enumerate(numerical):
            if nm in ['goose_size', 'longuest_opponent']:
                numerical_vector[i] = (float(step['cur_state'][nm])-8)/16
            else:
                numerical_vector[i] = step['cur_state'][nm]
        food_position_vector = np.zeros(len(food_position))
        for i, nm in enumerate(food_position):
            food_position_vector[i] = step['cur_state'][nm]
        numerical_vector[len(numerical)] = float(step['cur_state']['goose_size'] - step['cur_state']['longuest_opponent'])/10
        step['numerical'] = numerical_vector
        step['food_position_vector'] = food_position_vector
    return None



def add_embeddings(steps):
    numerical = ['food1_col',
                'food2_col',
                'food1_row',
                'food2_row',
                'goose_size',
                'hunger',
                'step']
    for step in steps:
        #vector = np.zeros(7*11, dtype=int)
        vector = []
        board = step['cur_state']['board']
        for row in range(7):
            for col in range(11):
               #vector[11*row + col] =  np.array(board[row][col], dtype=int)
               vector.append(int(board[row][col]))
        step['embeddings'] = vector
    return None


def add_state_value(discount, steps):
    steps_back = steps[::-1]
    v_prime = 0
    for step in steps_back:
        v = step['reward'] + discount*v_prime
        v_prime = v
        step['v'] = v
    return None



def add_next_state(steps):
    nb_steps = len(steps)
    for i, step in enumerate(steps):
        if step['status'] == 'ACTIVE':
            step['next_embeddings'] = steps[i+1]['embeddings']
            step['next_food_position_vector'] = steps[i+1]['food_position_vector']
            step['next_numerical'] = steps[i+1]['numerical']
        else:
            step['next_embeddings'] = steps[i]['embeddings']
            step['next_food_position_vector'] = steps[i]['food_position_vector']
            step['next_numerical'] = steps[i]['numerical']


def process(discount, episodes):
    for episode in episodes:
        add_embeddings(episode)
        add_numerical(episode)
        add_state_value(discount, episode)
        add_next_state(episode)
    return None


def training_data(episodes):
    targets = []
    next_numerical = []
    next_embeddings = []
    reward = []
    done = []
    v = []
    actions = []
    numerical = []
    embeddings = []
    for episode in episodes:
        for step in episode:
            action = step['action']
            target = action_to_target(action)
            targets.append(target)
            num = step['numerical']
            emb = step['embeddings']
            next_num = step['next_numerical']
            next_emb = step['next_embeddings']
            numerical.append(num)
            embeddings.append(emb)
            next_numerical.append(next_num)
            next_embeddings.append(next_emb)
            actions.append(action)
            v.append(step['v'])
            done.append(step['done'])
            reward.append(step['reward'])

    target_reshaped = np.array(targets).reshape(-1, 4)
    e = [np.array(embeddings)[:, i].reshape(-1, 1) for i in range(7*11)]
    n = [np.array(numerical)[:, i].reshape(-1, 1) for i in range(5)]
    train = n+e

    e_next = [np.array(next_embeddings)[:, i].reshape(-1, 1) for i in range(7 * 11)]
    n_next = [np.array(next_numerical)[:, i].reshape(-1, 1) for i in range(5)]
    train_next = n_next + e_next

    # if not done:
    #     e_next = [np.array(next_embeddings)[:, i].reshape(-1, 1) for i in range(7*11)]
    #     n_next = [np.array(next_numerical)[:, i].reshape(-1, 1) for i in range(5)]
    #     train_next = n_next+e_next
    # else:
    #     train_next = None

    training_dict = {'state': train,
                     'action': actions,
                     'next_state': train_next,
                     'y': target_reshaped,
                     'reward': reward,
                     'v': v,
                     'done': done}
    return training_dict