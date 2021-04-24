from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, \
                                                                row_col, adjacent_positions, translate, min_distance


import numpy as np
import pickle
from random import choice

from copy import deepcopy
from helpers import *
class RuleBasedAgent:
    """
    Rule based agent -
    We will use this rule-based agent to collect state-space data and the actions to take.
    An initial neural network will be trained to learn this rule-based policy.
    The neural network will then be improved using RL methods.
    """

    def __init__(self):
        self.last_action = None
        self.last_heads_positions = []
        self.stateSpace = None

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
        observation = Observation(obs_dict)
        player_index = observation.index
        player_goose = observation.geese[player_index]
        if len(player_goose) > 0:

            cur_obs, heads_positions, last_actions = self.getStateSpace(obs_dict, config_dict)

            food1_col_feat = cur_obs['food1_col']
            food2_col_feat = cur_obs['food2_col']
            food1_row_feat = cur_obs['food1_row']
            food2_row_feat = cur_obs['food2_row']
            player_goose_len = cur_obs['goose_size']
            board = cur_obs['board']
            cur_obs['hunger'] = -1 + (float(obs_dict['step'] % 40) / 20)
            cur_obs['step'] = (float(obs_dict['step']) / 100) - 1

            self.stateSpace = cur_obs

            # Prioritize food that is closer
            if (abs(food1_row_feat) + abs(food1_col_feat)) <= (
                    abs(food2_row_feat) + abs(food2_col_feat)):
                p1_food_row_feat = food1_row_feat
                p1_food_col_feat = food1_col_feat
                p2_food_row_feat = food2_row_feat
                p2_food_col_feat = food2_col_feat
            else:
                p1_food_row_feat = food2_row_feat
                p1_food_col_feat = food2_col_feat
                p2_food_row_feat = food1_row_feat
                p2_food_col_feat = food1_col_feat

            action = None

            action_dict = {}

            # For each possible action, we create a value using the following logic:
            # Is action eligible? If yes, +10 000 points
            # Will the action kill us right away? if no, +1000 points
            # Is there a possibility that any other player
            # move to that same box at that same step? If no, +100 points
            # Is this action getting us closer to the nearest food? If yes, +10 points
            # Is this action getting us closer to the other food? If yes, +1 points

            # We then take the action with the most points (won't kill us and
            # brings us toward food if possible)

            hunger_boost = 1
            if player_goose_len == 1:
                if (40 - obs_dict['step'] % 40) < 20:
                    hunger_boost = 10
                if (40 - obs_dict['step'] % 40) < 6:
                    hunger_boost = 100
                elif (40 - obs_dict['step'] % 40) < 3:
                    hunger_boost = 1000

            action_dict[Action.WEST.name] = 0
            # Is action eligible?
            if (self.last_action is None) | (self.last_action != Action.EAST.name):
                action_dict[Action.WEST.name] += 1E7
            # Will the action kill us right away?
            if (board[3, 4] == 0) | (board[3, 4] == 15):
                action_dict[Action.WEST.name] += 1E6
            # Will the action kill us on the subsequent step?:
            if not ((board[2, 4] in list(range(6, 15))) & (board[3, 3] in list(range(6, 15))) & (
                    board[4, 4] in list(range(6, 15)))):
                action_dict[Action.WEST.name] += 1E5
            # Could the action kill us on the subsequent step? - is there a head nearby?
            if not ((board[2, 4] in list(range(1, 5))) | (board[3, 3] in list(range(1, 5))) | (
                    board[4, 4] in list(range(1, 5)))):
                action_dict[Action.WEST.name] += 1E4
            # Could the action kill us on the subsequent step? - is there a head further?
            if (not ((board[2, 3] in list(range(1, 5))) | (board[3, 2] in list(range(1, 5))) | (
                    board[4, 3] in list(range(1, 5))))):
                action_dict[Action.WEST.name] += 1E3
            # Is there a possibility that any other player
            # move to that same box at that same step?
            if (board[3, 3] in [0, 1, 15]) & (board[4, 4] in [0, 4, 15]) & (board[2, 4] in [0, 3, 15]):
                action_dict[Action.WEST.name] += 1E2
            # Is this action getting us closer to the nearest food?
            if p1_food_col_feat < 0:
                action_dict[Action.WEST.name] += 1E1 * hunger_boost
            # Is this action getting us closer to the other food?
            if p2_food_col_feat < 0:
                action_dict[Action.WEST.name] += 1E0 * hunger_boost

            action_dict[Action.EAST.name] = 0
            if (self.last_action is None) | (self.last_action != Action.WEST.name):
                action_dict[Action.EAST.name] += 1E7
            if (board[3, 6] == 0) | (board[3, 6] == 15):
                action_dict[Action.EAST.name] += 1E6
            # Will the action kill us on the subsequent step?:
            if not ((board[2, 6] in list(range(6, 15))) & (board[3, 7] in list(range(6, 15))) & (
                    board[4, 6] in list(range(6, 15)))):
                action_dict[Action.EAST.name] += 1E5
            # Could the action kill us on the subsequent step? - is there a head nearby?
            if not ((board[2, 6] in list(range(1, 5))) | (board[3, 7] in list(range(1, 5))) | (
                    board[4, 6] in list(range(1, 5)))):
                action_dict[Action.EAST.name] += 1E4
            # Could the action kill us on the subsequent step? - is there a head further?
            if (not ((board[2, 7] in list(range(1, 5))) | (board[3, 8] in list(range(1, 5))) | (
                    board[4, 7] in list(range(1, 5))))):
                action_dict[Action.EAST.name] += 1E3
            if (board[3, 7] in [0, 2, 15]) & (board[4, 6] in [0, 4, 15]) & (board[2, 6] in [0, 3, 15]):
                action_dict[Action.EAST.name] += 1E2
            if p1_food_col_feat > 0:
                action_dict[Action.EAST.name] += 1E1 * hunger_boost
            if p2_food_col_feat > 0:
                action_dict[Action.EAST.name] += 1E0 * hunger_boost

            action_dict[Action.NORTH.name] = 0
            if (self.last_action is None) | (self.last_action != Action.SOUTH.name):
                action_dict[Action.NORTH.name] += 1E7
            if (board[2, 5] == 0) | (board[2, 5] == 15):
                action_dict[Action.NORTH.name] += 1E6
            # Will the action kill us on the subsequent step?:
            if not ((board[2, 4] in list(range(6, 15))) & (board[2, 6] in list(range(6, 15))) & (
                    board[1, 5] in list(range(6, 15)))):
                action_dict[Action.NORTH.name] += 1E5
            # Will the action kill us on the subsequent step?  - is there a head nearby?
            if not ((board[2, 4] in list(range(1, 5))) | (board[2, 6] in list(range(1, 5))) | (
                    board[1, 5] in list(range(1, 5)))):
                action_dict[Action.NORTH.name] += 1E4
            # Could the action kill us on the subsequent step? - is there a head further?
            if (not ((board[1, 4] in list(range(1, 5))) | (board[0, 5] in list(range(1, 5))) | (
                    board[1, 6] in list(range(1, 5))))):
                action_dict[Action.NORTH.name] += 1E3
            if (board[1, 5] in [0, 3, 15]) & (board[2, 4] in [0, 1, 15]) & (board[2, 6] in [0, 2, 15]):
                action_dict[Action.NORTH.name] += 1E2
            if p1_food_row_feat < 0:
                action_dict[Action.NORTH.name] += 1E1 * hunger_boost
            if p2_food_row_feat < 0:
                action_dict[Action.NORTH.name] += 1E0 * hunger_boost

            action_dict[Action.SOUTH.name] = 0
            if (self.last_action is None) | (self.last_action != Action.NORTH.name):
                action_dict[Action.SOUTH.name] += 1E7
            if (board[4, 5] == 0) | (board[4, 5] == 15):
                action_dict[Action.SOUTH.name] += 1E6
            # Will the action kill us on the subsequent step?:
            if not ((board[4, 4] in list(range(6, 15))) & (board[4, 6] in list(range(6, 15))) & (
                    board[5, 5] in list(range(6, 15)))):
                action_dict[Action.SOUTH.name] += 1E5
            # Will the action kill us on the subsequent step? - is there a head nearby?
            if not ((board[4, 4] in list(range(1, 5))) | (board[4, 6] in list(range(1, 5))) | (
                    board[5, 5] in list(range(1, 5)))):
                action_dict[Action.SOUTH.name] += 1E4
            # Could the action kill us on the subsequent step? - is there a head further?
            if (not ((board[5, 4] in list(range(1, 5))) | (board[6, 5] in list(range(1, 5))) | (
                    board[5, 6] in list(range(1, 5))))):
                action_dict[Action.SOUTH.name] += 1E3
            if (board[5, 5] in [0, 4, 15]) & (board[4, 4] in [0, 1, 15]) & (board[4, 6] in [0, 2, 15]):
                action_dict[Action.SOUTH.name] += 1E2
            if p1_food_row_feat > 0:
                action_dict[Action.SOUTH.name] += 1E1 * hunger_boost
            if p2_food_row_feat > 0:
                action_dict[Action.SOUTH.name] += 1E0 * hunger_boost

            west = np.sum(np.isin(board[1:6, 3:6], [0, 15]))
            east = np.sum(np.isin(board[1:6, 5:8], [0, 15]))
            north = np.sum(np.isin(board[1:4, 3:8], [0, 15]))
            south = np.sum(np.isin(board[3:6, 3:8], [0, 15]))

            pop = np.array([south, north, east, west])
            min_pop = np.argmax(pop)
            nb_empty = np.max(pop)

            actions = [Action.SOUTH.name, Action.NORTH.name, Action.EAST.name, Action.WEST.name]

            #         if nb_empty <= 10:
            #             action_dict[actions[min_pop]] += 10000
            #         if player_goose_len > 10:
            #             action_dict[actions[min_pop]] += 1000
            #         if player_goose_len > 1:
            #             action_dict[actions[min_pop]] += 100

            values = np.array([action_dict[action] for action in actions])

            max_equality = (values == max(values))
            possible_actions = (values * max_equality)
            action_pick = np.random.multinomial(1, possible_actions / sum(possible_actions))
            action = actions[np.argmax(action_pick)]
        else:
            action = Action.NORTH.name
            heads_positions = None

        self.last_action = action
        self.last_heads_positions = heads_positions
        return action