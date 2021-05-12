from tensorflow.keras.layers import Dense, Input, Embedding, Flatten, concatenate
from tensorflow.keras import Model, Sequential
import tensorflow as tf

from utils.state_space import *
from utils.helpers import pred_to_action, pred_to_action_greedy, target_to_action
from tensorflow.python.framework.ops import disable_eager_execution

import pickle
disable_eager_execution()

class ShortSightAgent:
    """
    Baseline agent to test reinforce algorithm - state space is only the last action taken,
    with the expectation that it will learn not to take opposite actions
    """

    def __init__(self, greedy=False, learning_rate=0.01):
        self.last_action = 'NORTH'
        self.stateSpace = None
        self.model = self._build_model(learning_rate)
        self.greedy = greedy

    def getStateSpace(self, obs_dict, last_action):
        self.last_action = last_action
        board, numerical = get_state_space(obs_dict, self.last_action, 1, True)
        return board, numerical

    def _build_model(self, lr):

        numerical = Input(shape=(8,))
        embedding = Embedding(2, 1, input_length=8)
        m = Sequential()
        m.add(embedding)
        m.add(Flatten())

        inputs = [numerical, m.input]
        outputs = [numerical] + m.outputs

        c = concatenate(outputs)
        pred = Dense(4, activation='linear')(c)

        G = Input(shape=(1, ))
        G_input = [G]

        def custom_loss(y_true, y_pred, G):
            log_softmax = tf.math.log_softmax(y_pred, axis=1)
            selected_action = tf.math.reduce_sum(tf.math.multiply(y_true, log_softmax), axis=1)
            selected_action_weighted = tf.math.multiply(selected_action, G)
            J = tf.math.reduce_mean(selected_action_weighted)
            l = -J
            return l

        def reinforce_loss(y_true, y_pred):
            return custom_loss(y_true, y_pred, G)

        cur_loss = reinforce_loss

        m = Model([inputs] + G_input, pred)

        optimizer = tf.keras.optimizers.Adam(lr=lr)
        m.compile(optimizer=optimizer,
                  loss=cur_loss,
                  metrics=[],
                  experimental_run_tf_function=False)
        return m

    def fit(self, X, y, val_X, val_y, batch_size=32, epoch=100):
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)
        self.model.fit(X,
                       y,
                       validation_data=(val_X, val_y),
                       epochs=epoch,
                       batch_size=batch_size,
                       callbacks=[callback])

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
        board, numerical = get_state_space(obs_dict, self.last_action, 1, True)

        self.stateSpace = board, numerical

        pred = self.model.predict([numerical.reshape(-1, 8), board.reshape(-1, 8),
                                   np.array([-1]).reshape(-1)])[0].astype('float64')
        if self.greedy:
            action = pred_to_action_greedy(pred)
        else:
            action = pred_to_action(pred)

        self.last_action = action

        return action
