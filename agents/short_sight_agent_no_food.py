from tensorflow.keras.layers import Dense, Input, Embedding, Flatten, concatenate
from tensorflow.keras import Model, Sequential
import tensorflow as tf

from utils.state_space import *
from utils.helpers import pred_to_action, pred_to_action_greedy, target_to_action
from tensorflow.python.framework.ops import disable_eager_execution

import pickle
disable_eager_execution()

class ShortSightAgentNoFood:
    """
    Baseline agent to test reinforce algorithm - state space is only the last action taken,
    with the expectation that it will learn not to take opposite actions
    """

    def __init__(self, greedy=False, learning_rate=0.01, entropy_reg=0):
        self.last_action = ['NORTH', 'EAST', 'WEST', 'SOUTH'][np.random.randint(4)]
        self.stateSpace = None
        self.model = self._build_model(learning_rate, entropy_reg)
        self.greedy = greedy

    def getStateSpace(self, obs_dict, last_action):
        self.last_action = last_action
        board, forbidden_action, food_pos = get_state_space(obs_dict, self.last_action, 2, True)
        return board, forbidden_action, food_pos

    def _build_model(self, lr, entropy_reg):

        forbidden_action = Input(shape=(4,))
        food_pos = Input(shape=(4,))
        embedding = Embedding(4, 1, input_length=24)
        m = Sequential()
        m.add(embedding)
        m.add(Flatten())

        #m.add(Dense(10, activation='elu'))

        food_m = Sequential()
        food_m.add(food_pos)
        #food_m.add(Dense(4, activation='elu'))

        concat = concatenate(m.outputs + food_m.outputs)
        #d = Dense(10, activation='elu')(concat)
        logits = Dense(4, activation='linear')(concat)

        # m.add(Dense(4, activation='linear'))

        inputs = [forbidden_action, m.input, food_m.inputs]

        # c = concatenate(outputs)
        # pred = Dense(4, activation='linear')(c)
        no_action = tf.math.multiply(forbidden_action, -10000)
        pred = tf.math.add(logits, no_action)

        G = Input(shape=(1, ))
        G_input = [G]

        def custom_loss(y_true, y_pred, G, numerical):
            log_softmax = tf.math.log_softmax(y_pred, axis=1)
            selected_action = tf.math.multiply(y_true, log_softmax)
            selected_action_weighted = tf.math.multiply(selected_action, G)
            # selected_action = tf.math.reduce_sum(tf.math.multiply(y_true, log_softmax), axis=1)
            # selected_action_weighted = tf.math.multiply(tf.reshape(selected_action, [-1]),
            #                                             tf.reshape(G, [-1]))
            possible_actions = tf.ones(shape=tf.shape(numerical)) - numerical
            softmax = tf.math.softmax(y_pred)
            entropy = -tf.reduce_mean(tf.math.multiply(tf.math.multiply(log_softmax, softmax),
                                                       possible_actions))
            J = tf.math.reduce_mean(selected_action_weighted) + entropy_reg*entropy
            l = -J
            return l

        def reinforce_loss(y_true, y_pred):
            return custom_loss(y_true, y_pred, G, forbidden_action)

        cur_loss = reinforce_loss

        m = Model([inputs] + G_input, pred)

        optimizer = tf.keras.optimizers.Adam(lr=lr)
        m.compile(optimizer=optimizer,
                  loss=cur_loss,
                  metrics=[],
                  experimental_run_tf_function=False)
        return m

    def fit(self, X, y, val_X, val_y, batch_size=32, epoch=2):
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

        self.model.fit(X,
                       y,
                       validation_data=(val_X, val_y) if len(val_X[0]) > 0 else None,
                       epochs=epoch if len(val_X[0]) > 0 else epoch,
                       batch_size=batch_size,
                       callbacks=[callback] if len(val_X[0]) > 0 else None)

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
        board, forbidden_action, food_pos = get_state_space(obs_dict, self.last_action, 2, True)

        self.stateSpace = board, forbidden_action, food_pos

        pred = self.model.predict([forbidden_action.reshape(-1, 4),
                                   board.reshape(-1, 24),
                                   food_pos.reshape(-1, 4),
                                   np.array([-1]).reshape(-1)])[0].astype('float64')
        if self.greedy:
            action = pred_to_action_greedy(pred)
        else:
            action = pred_to_action(pred)

        self.last_action = action

        return action
