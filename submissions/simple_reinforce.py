import numpy as np
from kaggle_environments.envs.hungry_geese.hungry_geese import Action

actions_dict = {
    Action.WEST.name: 0,
    Action.EAST.name: 1,
    Action.NORTH.name: 2,
    Action.SOUTH.name: 3,
}

actions_dict_invert = {v: k for k, v in actions_dict.items()}
actions_list = [actions_dict_invert[i] for i in range(4)]


def action_to_target(action):
    pos = actions_dict[action]
    target = np.zeros(4)
    target[pos] = 1
    return target


def target_to_action(target):
    pos = np.argmax(target)
    return actions_list[pos]


def softmax(x):
    z = x - np.max(x)
    return np.exp(z) / np.sum(np.exp(z))


def pred_to_action(pred, logit=True):
    if logit:
        pred = softmax(pred)
    pos = np.argmax(np.random.multinomial(1, pred))
    return actions_list[pos]


def pred_to_action_greedy(pred):
    pos = np.argmax(pred)
    return actions_list[pos]
from kaggle_environments.envs.hungry_geese.hungry_geese import row_col, Observation, Action
import numpy as np

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
            top = board[2:3, 4:7].reshape(-1)
            right = board[2:5, 6:7].reshape(-1)
            bottom = board[4:5, 6:3:-1].reshape(-1)
            left = board[4:1:-1, 4:5].reshape(-1)
            far_right = None
            far_left = None
        if size == 2:
            top = board[1:3, 3:8].reshape(-1)
            right = board[1:6, 7:5:-1].T.reshape(-1)
            bottom = board[5:3:-1, 7:2:-1].reshape(-1)
            left = board[5:0:-1, 3:5].T.reshape(-1)
            far_right = None
            far_left = None
        if size == 3:
            top = board[0:3, 2:9].reshape(-1)
            right = board[0:7, 8:5:-1].T.reshape(-1)
            bottom = board[6:3:-1, 8:1:-1].reshape(-1)
            left = board[np.array([6, 5, 4, 3, 2, 1, 0]), 2:5].T.reshape(-1)
            far_right = None
            far_left = None
        if size == 4:
            top = board[0:3, 2:9].reshape(-1)
            right = board[0:7, 8:5:-1].T.reshape(-1)
            bottom = board[6:3:-1, 8:1:-1].reshape(-1)
            left = board[np.array([6, 5, 4, 3, 2, 1, 0]), 2:5].T.reshape(-1)
            far_right = board[0:7, 10:8:-1].T.reshape(-1)
            far_left = board[np.array([6, 5, 4, 3, 2, 1, 0]), 0:2].T.reshape(-1)
        return top, right, bottom, left, far_right, far_left, board
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


from tensorflow.keras.layers import Dense, Input, Embedding, Flatten, concatenate
from tensorflow.keras import Model, Sequential
import tensorflow as tf
from tensorflow.keras.constraints import non_neg, UnitNorm

import pickle

class ShortSightAgentNoFood:
    """
    Baseline agent to test reinforce algorithm - state space is only the last action taken,
    with the expectation that it will learn not to take opposite actions
    """

    def __init__(self, greedy=True, learning_rate=0.01, entropy_reg=0):
        self.last_action = ['NORTH', 'EAST', 'WEST', 'SOUTH'][np.random.randint(4)]
        self.stateSpace = None
        self.model = self._build_model(learning_rate, entropy_reg)
        self.greedy = greedy

        serialized = b'(lp0\ncnumpy.core.multiarray\n_reconstruct\np1\n(cnumpy\nndarray\np2\n(I0\ntp3\nc_codecs\nencode\np4\n(Vb\np5\nVlatin1\np6\ntp7\nRp8\ntp9\nRp10\n(I1\n(I5\nI1\ntp11\ncnumpy\ndtype\np12\n(Vf4\np13\nI00\nI01\ntp14\nRp15\n(I3\nV<\np16\nNNNI-1\nI-1\nI0\ntp17\nbI00\ng4\n(V\xef\xb3\xb5>\x0e\xa0v\xc0\xde\xee\xb4\xbf\xb6~\x07\xc0-<O@\np18\ng6\ntp19\nRp20\ntp21\nbag1\n(g2\n(I0\ntp22\ng8\ntp23\nRp24\n(I1\n(I21\nI1\ntp25\ng15\nI00\ng4\n(V36\x98\xbe\x08\xa1u\xbeF\xa0\x12@\xde\xc7\xa7>\xd1\xd0\x15@\x18s\xba>\x87X\xaf\xbf\x02\xbc\xaa\xbf\x01\x17\xd1?\xba\xb0\xf4?\xba\xee\xcd@z\x91g?\xfe\xd8\x17@\x02\xea\x95\xbf\xb4Y\x0e?\xf62\x90?\xea\xa9\xa3@\xd1\xdd\x03AnS\xb2@\x1e\xc6\x8d><St\xbf\np26\ng6\ntp27\nRp28\ntp29\nbag1\n(g2\n(I0\ntp30\ng8\ntp31\nRp32\n(I1\n(I14\nI1\ntp33\ng15\nI00\ng4\n(Vq\xc2\x02\xbf\x81\x8b\xad>}\x82z?\xddY\x1f\xbeLh\xc2\xbeA_%\xbf{tX>\x7f\x05\x0c\xbf\xd7\xc8\x17\xbf\xa7\xa3\xa8>(\\u000d\xfc>\xe3\x80d\xbf*\x92B?\x1e`\xb0>\np34\ng6\ntp35\nRp36\ntp37\nbag1\n(g2\n(I0\ntp38\ng8\ntp39\nRp40\n(I1\n(I2\nI1\ntp41\ng15\nI00\ng4\n(V<\xd4\x90?\x1d\xb2D\xbe\np42\ng6\ntp43\nRp44\ntp45\nba.'
        weights = pickle.loads(serialized)
        self.model.set_weights(weights)

    def getStateSpace(self, obs_dict, last_action):
        self.last_action = last_action
        board, forbidden_action, food_pos = get_state_space(obs_dict, self.last_action, 4, False)
        return board, forbidden_action, food_pos

    def _build_model(self, lr, entropy_reg):
        tf.compat.v1.reset_default_graph()

        forbidden_action = Input(shape=(4,))

        norm_layer = UnitNorm(axis=1)
        embedding = Embedding(5, 1, input_length=21, trainable=True)

        top = Input(shape=(21,))
        right = Input(shape=(21,))
        bottom = Input(shape=(21,))
        left = Input(shape=(21,))

        far_right = Input(shape=(14,))
        far_left = Input(shape=(14,))

        common_linear = Dense(1, activation='linear', use_bias=False, trainable=True)

        far_sides_linear = Dense(1, activation='linear', use_bias=False, trainable=True)
        near_far_weighted = Dense(1, activation='linear', use_bias=False, trainable=True)
        paddings = tf.constant([[0, 0, ], [0, 7]])

        def far_sides(input):
            out = tf.pad(input, paddings, "CONSTANT")
            out = embedding(out)
            out = norm_layer(out)
            out = Flatten()(out)
            out, _ = tf.split(out, [14, 7], 1)
            out = far_sides_linear(out)
            return out

        def common_blocks(input):
            out = embedding(input)
            out = norm_layer(out)
            out = Flatten()(out)
            out = common_linear(out)
            return out

        def apply_side_layers(near, far):
            out_near = common_blocks(near)
            out_far = far_sides(far)
            concat = concatenate([out_near, out_far])
            out = near_far_weighted(concat)
            return out

        top_output = common_blocks(top)
        right_output = apply_side_layers(right, far_right)
        bottom_output = common_blocks(bottom)
        left_output = apply_side_layers(left, far_left)

        logits = concatenate([left_output, right_output, top_output, bottom_output])

        inputs = [forbidden_action, top, right, bottom, left, far_right, far_left]

        no_action = tf.math.multiply(forbidden_action, -10000)
        pred = tf.math.add(logits, no_action)


        m = Model([inputs] , pred)

        return m

    def __call__(self, obs_dict, config_dict):
        board, forbidden_action, food_pos = get_state_space(obs_dict, self.last_action, 4, False)

        self.stateSpace = board, forbidden_action, food_pos

        pred = self.model.predict([forbidden_action.reshape(-1, 4),
                                   board[0].reshape(-1, 21),
                                   board[1].reshape(-1, 21),
                                   board[2].reshape(-1, 21),
                                   board[3].reshape(-1, 21),
                                   board[4].reshape(-1, 14),
                                   board[5].reshape(-1, 14)])[0].astype('float64')
        if self.greedy:
            if board[6][3, 4] == 1:
                pred[0] = -1000
            if board[6][3, 4] == 2:
                pred[0] = -10000
            if board[6][3, 4] == 3:
                pred[0] = -500
            if board[6][3, 6] == 1:
                pred[1] = -1000
            if board[6][3, 6] == 2:
                pred[1] = -10000
            if board[6][3, 6] == 3:
                pred[1] = -500
            if board[6][2, 5] == 1:
                pred[2] = -1000
            if board[6][2, 5] == 2:
                pred[2] = -10000
            if board[6][2, 5] == 3:
                pred[2] = -500
            if board[6][4, 5] == 1:
                pred[3] = -1000
            if board[6][4, 5] == 2:
                pred[3] = -10000
            if board[6][4, 5] == 3:
                pred[3] = -500

            action = pred_to_action_greedy(pred)
        else:
            action = pred_to_action(pred)

        self.last_action = action

        return action


my_agent = ShortSightAgentNoFood()

def agent(obs_dict, config_dict):
    return my_agent(obs_dict, config_dict)