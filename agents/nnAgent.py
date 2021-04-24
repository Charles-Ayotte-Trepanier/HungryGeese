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
    def __init__(self, prediction_type):
        self.prediction_type = prediction_type
        self.model = None

    def build_model(self, embedding_size=2, all_trainable=True):

        if self.prediction_type == 'policy':
            activation = 'softmax'
            loss = 'categorical_crossentropy'
            metrics = ['accuracy', 'AUC']
            output_dim = 4
        elif self.prediction_type == 'win_prob':
            activation = 'sigmoid'
            loss = 'binary_crossentropy'
            metrics = ['accuracy', 'AUC']
            output_dim = 1
        elif self.prediction_type == 'state_value':
            activation = 'linear'
            loss = 'mean_absolute_error'
            metrics = ['mae']
            output_dim = 1
        elif self.prediction_type == 'actor_critic':
            activation = 'softmax'
            metrics = ['accuracy', 'AUC']
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

        final = Dense(output_dim, activation='linear')(model)
        act = 
        m = Model(inputs, model)

        if self.prediction_type == 'actor_critic':
            advantage = [Input(shape=1)]
            loss =
        else:
            advantage = []
        m = Model(inputs + advantage, model)

        m.compile(optimizer='adam',
                  loss=loss,
                  metrics=metrics)
        self.model = m

    def replace_weights(self, policy_model):
        policy_weights = policy_model.get_weights()
        cur_weights = self.model.get_weights()
        for i in range(20):
            cur_weights[i] = policy_weights[i]
        self.model.set_weights(cur_weights)

    def predict(self, X):
        return self.model.predict(X)

    def fit(self, X_train, y_train, X_test, y_test, epoch=1, batch_size=32):
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

        callbacks = [perf_callback(X_test, y_test, self.prediction_type ),
                     early_stop]

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
