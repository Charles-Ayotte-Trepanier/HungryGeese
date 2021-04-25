import os
from math import ceil
import numpy as np
from tensorflow.keras.layers import Dense,Input, Embedding, concatenate,\
    Flatten, Average, Dropout, BatchNormalization, Activation
from tensorflow.keras import Sequential, Model
from tensorflow import config, distribute
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import Callback, EarlyStopping
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.python.framework.ops import disable_eager_execution
import pickle

disable_eager_execution()

def train_test_splitter(X, y, test_ratio, v=None):
    train, test = train_test_split(range(y.shape[0]), test_size=test_ratio)
    X_train = [feature[train, :] for feature in X]
    X_test = [feature[test, :] for feature in X]
    y_train = y[train, :]
    y_test = y[test, :]
    add = ()
    if v is not None:
        v_train = np.array(v, dtype=float)[train].reshape(-1, 1)
        v_test = np.array(v, dtype=float)[test].reshape(-1, 1)
        add += (v_train, v_test)
    return (X_train, X_test, y_train, y_test) + add


class nnModel:
    def __init__(self, prediction_type, entropy_weight=0.1):
        self.prediction_type = prediction_type
        self.model = None
        self.entropy_weight = entropy_weight

    def build_model(self, embedding_size=2, all_trainable=True):

        if self.prediction_type == 'policy':
            activation = 'softmax'
            cur_loss = 'categorical_crossentropy'
            metrics = ['accuracy', 'AUC']
            output_dim = 4
        elif self.prediction_type == 'win_prob':
            activation = 'sigmoid'
            cur_loss = 'binary_crossentropy'
            metrics = ['accuracy', 'AUC']
            output_dim = 1
        elif self.prediction_type == 'state_value':
            activation = 'linear'
            cur_loss = 'mean_absolute_error'
            metrics = ['mae']
            output_dim = 1
        elif self.prediction_type == 'actor_critic':
            activation = 'softmax'
            metrics = []
            output_dim = 4

        embedding = Embedding(17, embedding_size, input_length=1, trainable=all_trainable)
        num_layers = []
        for _ in range(5):
            num_layers.append(Input(shape=1))
        emb_layers = []
        for i in range(7 * 11):
            m = Sequential()
            embedding._name = f'embeddings_{i}'
            m.add(embedding)
            m.add(Flatten(name=f'flat_embeddings-{i}'))
            emb_layers.append(m)

        inputs = num_layers + [inp.input for inp in emb_layers]
        outputs = num_layers + [inp.output for inp in emb_layers]

        c = concatenate(outputs)
        model = Dense(200, activation='elu', trainable=all_trainable)(c)
        model = BatchNormalization(trainable=all_trainable)(model)
        model = Dropout(rate=0.2, input_shape=(200,), trainable=all_trainable)(model)
        model = Dense(100, activation='elu', trainable=all_trainable)(model)
        model = Dropout(rate=0.2, input_shape=(100,), trainable=all_trainable)(model)
        model = BatchNormalization(trainable=all_trainable)(model)
        model = Dense(50, activation='elu', trainable=all_trainable)(model)
        model = BatchNormalization(trainable=all_trainable)(model)
        model = Dense(20, activation='elu')(model)
        model = BatchNormalization()(model)
        model = Dense(10, activation='elu')(model)

        #pred = Dense(output_dim, activation=activation)(model)

        logit = Dense(output_dim, activation='linear')(model)

        pred = Activation(activation=activation)(logit)
        if self.prediction_type == 'actor_critic':
            advantage = Input(shape=(1, ))
            advantage_input = [advantage]

            def custom_loss(y_true, y_pred, advantage, logit):
                # testing = y_pred + 0.001
                # y_pred = testing / tf.math.reduce_sum(testing)
                log_softmax = tf.math.log_softmax(logit)
                softmax = tf.math.softmax(logit)
                entropy = -tf.math.reduce_sum(log_softmax * softmax)
                selected_action = tf.math.reduce_sum(y_true*log_softmax, axis=0)
                selected_action_weighted = selected_action * advantage
                J = tf.math.reduce_mean(selected_action_weighted)
                entropy_weights = self.entropy_weight*entropy
                l = -(J + entropy_weights)
                return l

            def reinforce_loss(y_true, y_pred):
                return custom_loss(y_true, y_pred, advantage, logit)

            cur_loss = reinforce_loss

        else:
            advantage_input = []

        m = Model(inputs + advantage_input, pred)

        m.compile(optimizer='adam',
                  loss=cur_loss,
                  metrics=metrics,
                  experimental_run_tf_function=False)
        self.model = m

    def replace_partial_weights(self, policy_model):
        policy_weights = policy_model.model.get_weights()
        cur_weights = self.model.get_weights()
        for i in range(20):
            cur_weights[i] = policy_weights[i]
        self.model.set_weights(cur_weights)

    def replace_weights(self, other_model):
        self.model.set_weights(other_model.model.get_weights())

    def predict(self, X):
        if (len(X) == 82) and (self.prediction_type == 'actor_critic'): #missing advantage
            X = X + [np.zeros(shape=X[0].shape)]
        return self.model.predict(X)

    def fit(self, X_train, y_train, X_test=None, y_test=None, epoch=1, batch_size=32):
        def reset_weights(model):
            session = keras.get_session()
            for layer in model.layers:
                if hasattr(layer, 'kernel_initializer'):
                    layer.kernel.initializer.run(session=session)
                if hasattr(layer, 'bias_initializer'):
                    layer.bias.initializer.run(session=session)

        class perf_callback(Callback):
            def __init__(self, X, y, model_type):
                self.X = X
                self.y = y
                self.model_type = model_type

            def on_epoch_end(self, epoch, logs={}):
                y_pred = self.model.predict(self.X)

                if self.model_type == 'policy':
                    perf = roc_auc_score(self.y, y_pred, average='weighted')
                    accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(self.y, axis=1))
                    print(f'Validation auc: {perf}')
                    print(f'Validation accuracy: {accuracy}')
                elif self.model_type == 'win_prob':
                    perf = roc_auc_score(self.y, y_pred, average='average')
                    accuracy = np.mean((y_pred.reshape(-1) >=0.5) == (self.y.reshape(-1) == 1))
                    print(f'Validation auc: {perf}')
                    print(f'Validation accuracy: {accuracy}')
                elif self.model_type == 'state_value':
                    perf = r2_score(self.y, y_pred)
                    print(f'Validation r2: {perf}')
                logs['validation'] = perf

        early_stop = EarlyStopping(patience=2,
                                   monitor='validation',
                                   mode='max')

        if X_test is not None:
            callbacks = [perf_callback(X_test, y_test, self.prediction_type ),
                         early_stop]
        else:
            callbacks = []

        self.model.fit(X_train,
                       y_train,
                       epochs=epoch,
                       batch_size=batch_size,
                       callbacks=callbacks)

    def load(self, name):
        path = f'/home/charles/PycharmProjects/HungryGeese/models/{name}'
        self.model = keras.models.load_model(path)
        return self.model

    def save(self, name):
        path = f'/home/charles/PycharmProjects/HungryGeese/models/{name}'
        self.model.save(path)

    def save_weights(self, name):
        path = f'/home/charles/PycharmProjects/HungryGeese/models/{name}_weights.pkl'
        with open(path, 'wb') as f:
            pickle.dump(self.model.get_weights(), f)

    def load_weights(self, name):
        path = f'/home/charles/PycharmProjects/HungryGeese/models/{name}_weights.pkl'
        with open(path, 'rb') as f:
            weights = pickle.load(f)
        self.model.set_weights(weights)
