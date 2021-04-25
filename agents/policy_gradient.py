from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, \
                                                                row_col, adjacent_positions, translate, min_distance


import numpy as np
import pickle
from random import choice

from copy import deepcopy
from helpers import *

class PolicyModelAgent:
    """
    policy NN model
    """

    def __init__(self, model, greedy=False):
        self.last_action = None
        self.last_heads_positions = []
        self.stateSpace = None
        self.model = model
        self.greedy = greedy

    def getStateSpace(self, obs_dict, config_dict):
        heads_positions = geese_heads(obs_dict, config_dict)
        last_actions = get_last_actions(self.last_heads_positions, heads_positions)

        board, player_goose_len, longuest_opponent, food1_row_feat, food1_col_feat, food2_row_feat, food2_col_feat = central_state_space(
            obs_dict, config_dict, last_actions)

        cur_obs = {}
        cur_obs['food1_col'] = food1_col_feat
        cur_obs['food2_col'] = food2_col_feat
        cur_obs['food1_row'] = food1_row_feat
        cur_obs['food2_row'] = food2_row_feat
        cur_obs['goose_size'] = player_goose_len
        cur_obs['longuest_opponent'] = longuest_opponent
        cur_obs['board'] = board
        cur_obs['hunger'] = -1 + (float(obs_dict['step'] % 40) / 20)
        cur_obs['step'] = (float(obs_dict['step']) / 100) - 1

        return cur_obs, heads_positions, last_actions

    def __call__(self, obs_dict, config_dict):
        cur_obs, heads_positions, last_actions = self.getStateSpace(obs_dict, config_dict)

        player_goose_len = cur_obs['goose_size']
        board = cur_obs['board']
        cur_obs['hunger'] = -1 + (float(obs_dict['step'] % 40) / 20)
        cur_obs['step'] = (float(obs_dict['step']) / 100) - 1

        self.stateSpace = cur_obs

        my_size = np.array((float(player_goose_len) - 8) / 16).reshape(-1, 1)
        longest = np.array((float(cur_obs['longuest_opponent']) - 8) / 16).reshape(-1, 1)
        hunger = np.array(cur_obs['hunger']).reshape(-1, 1)
        cur_step = np.array(cur_obs['step']).reshape(-1, 1)
        diff = np.array(float(my_size - longest) / 10).reshape(-1, 1)

        obs = [my_size, longest, hunger, cur_step, diff]
        for row in range(7):
            for col in range(11):
                obs.append(np.array(int(board[row][col])).reshape(-1, 1))

        pred = self.model.predict(obs)[0].astype('float64')
        if self.greedy:
            action = pred_to_action_greedy(pred)
        else:
            action = pred_to_action(pred/np.sum(pred))

        return action