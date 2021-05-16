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

from kaggle_environments.envs.hungry_geese.hungry_geese import row_col, Observation, Action
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
        distances = np.array([5, 5, 5, 4, 5, 5, 5,
                              5, 5, 4, 3, 4, 5, 5,
                              5, 4, 3, 2, 3, 4, 5,
                              5, 3, 2, 1, 2, 3, 4,
                              3, 2, 1, 0, 1, 2, 3]).reshape(5, 7)
        return top, right, bottom, left, bodies, distances, board
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

from tensorflow.keras.layers import Dense, Input, Embedding, Flatten, concatenate, Conv2D
from tensorflow.keras import Model, Sequential
import tensorflow as tf
from tensorflow.keras.constraints import non_neg, UnitNorm

from tensorflow.python.framework.ops import disable_eager_execution

import pickle


class CnnAgent:
    """
    Baseline agent to test reinforce algorithm - state space is only the last action taken,
    with the expectation that it will learn not to take opposite actions
    """

    def __init__(self, greedy=False, learning_rate=0.01, entropy_reg=0):
        self.last_action = ['NORTH', 'EAST', 'WEST', 'SOUTH'][np.random.randint(4)]
        self.stateSpace = None
        self.model = self._build_model(learning_rate, entropy_reg)
        self.greedy = greedy
        serialized = b'(lp0\ncnumpy.core.multiarray\n_reconstruct\np1\n(cnumpy\nndarray\np2\n(I0\ntp3\nc_codecs\nencode\np4\n(Vb\np5\nVlatin1\np6\ntp7\nRp8\ntp9\nRp10\n(I1\n(I5\nI1\ntp11\ncnumpy\ndtype\np12\n(Vf4\np13\nI00\nI01\ntp14\nRp15\n(I3\nV<\np16\nNNNI-1\nI-1\nI0\ntp17\nbI00\ng4\n(V\xb0>\x0b\xbf\xbbE\xaf@m\x88P@c\xbc\x0e?\xf4\x9f6\xc0\np18\ng6\ntp19\nRp20\ntp21\nbag1\n(g2\n(I0\ntp22\ng8\ntp23\nRp24\n(I1\n(I6\nI1\ntp25\ng15\nI00\ng4\n(V\x11?\x90@.\x8ew@z\xd1X>\xd8d\x19?6x5\xbfV\xb6\xf3?\np26\ng6\ntp27\nRp28\ntp29\nbag1\n(g2\n(I0\ntp30\ng8\ntp31\nRp32\n(I1\n(I3\nI3\nI1\nI2\ntp33\ng15\nI00\ng4\n(V\xa5\xa1"?\xdf\x08D\xbf\x93\x10 \xbf\xb4\xc5\x93\xbe&+\xe5>R\xfc[\xbfxBk\xbe\xee!\x91\xbe\x98:\xef\xbe\x13\x95\t\xbf\xa9\xfbI\xbf\xefk@\xbf\xdbs\x07>\xe9&\xc7\xbe\xf3G|\xbe\xa6\xb3\xd0\xbej\xcb\x86\xbe\x18j\x0b\xbf\np34\ng6\ntp35\nRp36\ntp37\nbag1\n(g2\n(I0\ntp38\ng8\ntp39\nRp40\n(I1\n(I2\ntp41\ng15\nI00\ng4\n(VY\x17\x99\xbe\x1c\xc1Y=\np42\ng6\ntp43\nRp44\ntp45\nbag1\n(g2\n(I0\ntp46\ng8\ntp47\nRp48\n(I1\n(I2\nI2\nI2\nI2\ntp49\ng15\nI00\ng4\n(VI \x84>\xcbp\xe5=\xdd\x08\xd1\xbe\x94\xb9\xc5;/\xd6F?\xd0\x8af=\xaa]_\xbf\xe2\xeb=>\x1bSv\xbf\x84\xa0s?H\xf3K\xbf\xe0\\u000d*?4\x8b\x97\xbe\x17\xec\xde=\x8d\x87\x8c\xbcDQ ?\np50\ng6\ntp51\nRp52\ntp53\nbag1\n(g2\n(I0\ntp54\ng8\ntp55\nRp56\n(I1\n(I2\ntp57\ng15\nI00\ng4\n(Vh\x88\xf2\xbe\xd0\xf8\x90\xbe\np58\ng6\ntp59\nRp60\ntp61\nbag1\n(g2\n(I0\ntp62\ng8\ntp63\nRp64\n(I1\n(I2\nI2\nI2\nI1\ntp65\ng15\nI00\ng4\n(V\x9f\xbe\xcf\xbe\xfc0`\xbf\x83\x13r\xbf ;\xf7=%n\x91>\xa0\xfc\xc2\xbe\xf6"\xe7>o\xd1\x85\xbf\np66\ng6\ntp67\nRp68\ntp69\nbag1\n(g2\n(I0\ntp70\ng8\ntp71\nRp72\n(I1\n(I1\ntp73\ng15\nI00\ng4\n(V\xee\x19\xf6>\np74\ng6\ntp75\nRp76\ntp77\nbag1\n(g2\n(I0\ntp78\ng8\ntp79\nRp80\n(I1\n(I3\nI4\ntp81\ng15\nI00\ng4\n(V\x05\xb5\x91\xbf\xbe\x9b\x96?\xbbc\xd4\xbd\\u001a%\x19?[.\\u000d\xbfnO\x89?ih\xba>\xa0}b>F\x1e\x15>\x83_r?&\xdc\xb7\xbf8(*\xbe\np82\ng6\ntp83\nRp84\ntp85\nbag1\n(g2\n(I0\ntp86\ng8\ntp87\nRp88\n(I1\n(I4\ntp89\ng15\nI00\ng4\n(V\x14z$>H\xc3\xfa>\x1b\xa6t>|b\x95\xbe\np90\ng6\ntp91\nRp92\ntp93\nbag1\n(g2\n(I0\ntp94\ng8\ntp95\nRp96\n(I1\n(I4\nI1\ntp97\ng15\nI00\ng4\n(Vg\xf9\x84?\xa7w\x87\xbf\xd7\t\x8a>\x979\x1b\xbe\np98\ng6\ntp99\nRp100\ntp101\nba.'
        weights = pickle.loads(serialized)
        self.model.set_weights(weights)

    def getStateSpace(self, obs_dict, last_action):
        self.last_action = last_action
        board, forbidden_action, food_pos = get_state_space(obs_dict, self.last_action, 5, False)
        return board, forbidden_action, food_pos

    def _build_model(self, lr, entropy_reg):

        forbidden_action = Input(shape=(4,))
        bodies = Input(shape=(4,))

        norm_layer = UnitNorm(axis=1)
        embedding = Embedding(5, 1, input_length=35, trainable=True)
        distance_embeddings = Embedding(6, 1, input_length=35, trainable=True)



        top = Input(shape=(35,))
        right = Input(shape=(35,))
        bottom = Input(shape=(35,))
        left = Input(shape=(35,))

        distance = Input(shape=(35,))
        embedded_distances = Flatten()(distance_embeddings(distance))

        cnn1 = Conv2D(2, 3, activation='elu', trainable=True)
        cnn2 = Conv2D(2, 2, activation='elu', trainable=True)
        cnn3 = Conv2D(1, 2, activation='elu', trainable=True)
        dense = Dense(4, activation='elu', trainable=True)
        linear = Dense(1, activation='linear', use_bias=False, trainable=True)

        def common_blocks(input):
            out = embedding(input)
            out = norm_layer(out)
            out = tf.math.multiply(Flatten()(out), embedded_distances)
            out = tf.reshape(out, [-1, 5, 7, 1])
            out = cnn1(out)
            out = cnn2(out)
            out = cnn3(out)
            out = Flatten()(out)
            out = dense(out)
            out = linear(out)
            return out

        top_output = common_blocks(top)
        right_output = common_blocks(right)
        bottom_output = common_blocks(bottom)
        left_output = common_blocks(left)

        logits = concatenate([left_output, right_output, top_output, bottom_output])

        inputs = [forbidden_action, top, right, bottom, left, bodies, distance]

        no_action = tf.math.multiply(forbidden_action, -10000)
        #no_action2 = tf.math.multiply(bodies, -10000)
        pred = tf.math.add(logits, no_action)
        #pred = tf.math.add(pred, no_action2)

        G = Input(shape=(1, ))
        G_input = [G]

        def custom_loss(y_true, y_pred, G, forbidden_action, bodies):
            log_softmax = tf.math.log_softmax(y_pred, axis=1)
            selected_action = tf.math.multiply(y_true, log_softmax)
            selected_action_weighted = tf.math.multiply(selected_action, G)
            # selected_action = tf.math.reduce_sum(tf.math.multiply(y_true, log_softmax), axis=1)
            # selected_action_weighted = tf.math.multiply(tf.reshape(selected_action, [-1]),
            #                                             tf.reshape(G, [-1]))
            # no_go = tf.math.maximum(forbidden_action, bodies)
            no_go = forbidden_action
            possible_actions = tf.ones(shape=tf.shape(forbidden_action)) - no_go
            softmax = tf.math.softmax(y_pred)
            entropy = -tf.reduce_mean(tf.math.multiply(tf.math.multiply(log_softmax, softmax),
                                                       possible_actions))
            J = tf.math.reduce_mean(selected_action_weighted) + entropy_reg*entropy
            l = -J
            return l

        def reinforce_loss(y_true, y_pred):
            return custom_loss(y_true, y_pred, G, forbidden_action, bodies)

        cur_loss = reinforce_loss

        m = Model([inputs] + G_input, pred)

        optimizer = tf.keras.optimizers.Adam(lr=lr)
        m.compile(optimizer=optimizer,
                  loss=cur_loss,
                  metrics=[],
                  experimental_run_tf_function=False)
        return m

    def save_weights(self, name):
        path = f'/home/charles/PycharmProjects/HungryGeese/models/{name}_weights.pkl'
        with open(path, 'wb') as f:
            pickle.dump(self.model.get_weights(), f)

    def load_weights(self, name):
        path = f'/home/charles/PycharmProjects/HungryGeese/models/{name}_weights.pkl'
        with open(path, 'rb') as f:
            weights = pickle.load(f)
        self.model.set_weights(weights)

    def __call__(self, obs_dict, config_dict):
        board, forbidden_action, food_pos = get_state_space(obs_dict, self.last_action, 5, False)

        self.stateSpace = board, forbidden_action, food_pos

        pred = self.model.predict([forbidden_action.reshape(-1, 4),
                                   board[0].reshape(-1, 35),
                                   board[1].reshape(-1, 35),
                                   board[2].reshape(-1, 35),
                                   board[3].reshape(-1, 35),
                                   board[4].reshape(-1, 4),
                                   board[5].reshape(-1, 35),
                                   np.array([-1]).reshape(-1)])[0].astype('float64')
        if self.greedy:
            if board[6][3, 4] == 2:
                pred[0] = -10000
            if board[6][3, 6] == 2:
                pred[1] = -10000
            if board[6][2, 5] == 2:
                pred[2] = -10000
            if board[6][4, 5] == 2:
                pred[3] = -10000

        else:
            action = pred_to_action(pred)

        self.last_action = action

        return action


my_agent = CnnAgent()


def agent(obs_dict, config_dict):
    return my_agent(obs_dict, config_dict)