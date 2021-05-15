from kaggle_environments.envs.hungry_geese.hungry_geese import row_col, Observation, Action
import numpy as np
from utils.helpers import actions_dict

def get_encoding(part, food_vector):
    encodings = {
        'head': 1,
        'body': 2,
        'tail': 3,
        'food': 4
    }

    if food_vector:
        encodings['food'] = 0
    return encodings[part]


def goose_encodings(position, goose_length, food_vector):

    if position == 0:
        part = 'head'
    elif position == goose_length - 1:
        part = 'tail'
    else:
        part = 'body'
    return get_encoding(part, food_vector)

def get_coordinate(position):
    return row_col(position, 11)


def create_mapping(head_position):
    player_row, player_column = get_coordinate(head_position)

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

    position_dict = {}
    for pos in range(7*11):
        i, j = get_coordinate(pos)
        position_dict[pos] = centralize(i, j)

    return position_dict


class StateSpace:
    def __init__(self):
        self._mapping = self.generate_mappings()

    @staticmethod
    def generate_mappings():
        mapping = {}
        for head in range(7*11):
            mapping[head] = create_mapping(head)
        return mapping

    def get_position(self, reference, other):
        return self._mapping[reference][other]


class FeaturesCreator:
    def __init__(self):
        self.get_position = StateSpace().get_position

    def get_features(self, obs_dict, last_action, board_size=0, food_vector=True):
        forbidden_action_vector = self._get_forbidden_action(last_action)
        if board_size > 0:
            board = self._get_board_sections(obs_dict, board_size, food_vector)
        else:
            board = np.array([])

        if food_vector:
            food_features = self._get_food_features(obs_dict)
        else:
            food_features = np.array([])

        return board, forbidden_action_vector, food_features

    def _get_last_action(self, last_action):
        actions = np.zeros(4)
        actions[actions_dict[last_action]] = 1
        return actions

    def _get_forbidden_action(self, last_action):
        actions = np.zeros(4)
        if last_action == 'NORTH':
            forbidden_action = 'SOUTH'
        if last_action == 'SOUTH':
            forbidden_action = 'NORTH'
        if last_action == 'WEST':
            forbidden_action = 'EAST'
        if last_action == 'EAST':
            forbidden_action = 'WEST'
        actions[actions_dict[forbidden_action]] = 1
        return actions

    def _get_food_features(self, obs_dict):
        observation = Observation(obs_dict)
        player_index = observation.index
        player_goose = observation.geese[player_index]
        player_head = player_goose[0]
        food_positions = observation.food

        if len(food_positions) == 2:
            food1_row, food1_column = self.get_position(player_head, food_positions[0])
            food2_row, food2_column = self.get_position(player_head, food_positions[1])
        elif len(food_positions) == 1:
            food1_row, food1_column = self.get_position(player_head, food_positions[0])
            food2_row, food2_column = self.get_position(player_head, player_head)
        elif len(food_positions) == 0:
            food1_row, food1_column = self.get_position(player_head, player_head)
            food2_row, food2_column = self.get_position(player_head, player_head)

        if food1_row != 3:
            food1_row_feat = 1.0/float(food1_row - 3)
        else:
            food1_row_feat = 0
        if food2_row != 3:
            food2_row_feat = 1.0/float(food2_row - 3)
        else:
            food2_row_feat = 0
        if food1_column != 5:
            food1_col_feat = 1.0/float(food1_column - 5)
        else:
            food1_col_feat = 0
        if food2_column != 5:
            food2_col_feat = 1.0 / float(food2_column - 5)
        else:
            food2_col_feat = 0

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

        return np.array([p1_food_row_feat, p1_food_col_feat, p2_food_row_feat, p2_food_col_feat])

    def _get_board_sections(self, obs_dict, size, food_vector):
        board = self._get_board(obs_dict, food_vector)
        if size == 1:
            top = board[2:3, 4:7]
            right = board[2:5, 6:7]
            bottom = board[4:5, 6:3:-1]
            left = board[4:1:-1, 4:5]
        if size == 2:
            top = board[1:3, 3:8]
            right = board[1:6, 7:5:-1].T
            bottom = board[5:3:-1, 7:2:-1]
            left = board[5:0:-1, 3:5].T
        if size == 3:
            top = board[0:3, 2:9]
            right = board[0:7, 8:5:-1].T
            bottom = board[6:3:-1, 8:1:-1]
            left = board[np.array([6, 5, 4, 3, 2, 1, 0]), 2:5].T
        if size == 4:
            top = np.vstack((board[6:7, 2:9], board[0:3, 2:9]))
            right = board[0:7, 9:5:-1].T
            bottom = np.vstack((board[0, 8:1:-1], board[6:3:-1, 8:1:-1]))
            left = board[np.array([6, 5, 4, 3, 2, 1, 0]), 1:5].T
        if size == 5:
            top = np.vstack((board[5:7, 2:9], board[0:3, 2:9]))
            right = board[0:7, 10:5:-1].T
            bottom = np.vstack((board[np.array([1, 0]), 8:1:-1], board[6:3:-1, 8:1:-1]))
            left = board[np.array([6, 5, 4, 3, 2, 1, 0]), 0:5].T

        bodies = np.zeros(4)
        if board[3, 4] == 2:
            bodies[0] = 1
        if board[3, 6] == 2:
            bodies[1] = 1
        if board[2, 5] == 2:
            bodies[2] = 1
        if board[4, 5] == 2:
            bodies[3] = 1
        return top, right, bottom, left, bodies
    # def _get_board_section(self, obs_dict, size, food_vector):
    #     board = self._get_board(obs_dict, food_vector)
    #     if size == 1:
    #         features = board[2:5, 4:7].reshape(-1)
    #     elif size == 2:
    #         features = board[1:6, 3:8].reshape(-1)
    #     elif size == 3:
    #         features = board[:, 2:9].reshape(-1)
    #     elif size == 4:
    #         features = board.reshape(-1)
    #
    #     middle_index = int((len(features)-1)/2)
    #     features = np.delete(features, [middle_index])
    #
    #     return features

    def _get_board(self, obs_dict, food_vector):
        board = np.zeros(7*11).reshape(7, 11)
        observation = Observation(obs_dict)
        player_index = observation.index
        player_goose = observation.geese[player_index]
        player_head = player_goose[0]

        for goose in observation.geese:
            cur_goose_length = len(goose)
            for i, position in enumerate(goose):
                coordinates = self.get_position(player_head, position)
                value = goose_encodings(i, cur_goose_length, food_vector)
                board[coordinates] = value

        for food_pos in obs_dict['food']:
            coordinates = self.get_position(player_head, food_pos)
            value = get_encoding('food', food_vector)
            board[coordinates] = value

        return board

fs = FeaturesCreator()

def get_state_space(obs_dict, last_action, board_size=0, food_vector=True):
    return fs.get_features(obs_dict, last_action, board_size, food_vector)