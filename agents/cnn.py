from tensorflow.keras.layers import Dense, Input, Embedding, Flatten, concatenate, Conv2D
from tensorflow.keras import Model, Sequential
import tensorflow as tf
from tensorflow.keras.constraints import non_neg, UnitNorm
from utils.state_space import *
from utils.helpers import pred_to_action, pred_to_action_greedy, target_to_action
from tensorflow.python.framework.ops import disable_eager_execution

import pickle
disable_eager_execution()

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

    def fit(self, X, y, val_X, val_y, batch_size=32, epoch=10):
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
            action = pred_to_action_greedy(pred)
        else:
            action = pred_to_action(pred)

        self.last_action = action

        return action
