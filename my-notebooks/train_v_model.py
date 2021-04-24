import os, pickle
from math import ceil
import numpy as np
from tensorflow.keras.layers import Dense,Input, Embedding, concatenate,\
    Flatten, Average, Dropout, BatchNormalization, Activation
from tensorflow.keras import Sequential, Model
from tensorflow import config, distribute
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import Callback, EarlyStopping
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split
def train_test_splitter(X, y, test_ratio, v=None):
    train, test = train_test_split(range(y.shape[0]), test_size=test_ratio)
    X_train = [feature[train,:] for feature in X]
    X_test = [feature[test,:] for feature in X]
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
        
    def build_model(self):
        embedding = Embedding(17, 2, input_length=1)
        num_layers = []
        for _ in range(5):
            num_layers.append(Input(shape=1))
        emb_layers = []
        for i in range(7*11):
            m = Sequential()
            embedding._name = f'embeddings_{i}'
            m.add(embedding)
            m.add(Flatten(name=f'flat_embeddings-{i}'))
            emb_layers.append(m)

        inputs = num_layers + [inp.input for inp in emb_layers]
        outputs = num_layers + [inp.output for inp in emb_layers]

        c = concatenate(outputs)
        model = Dense(200, activation='elu')(c)
        model = BatchNormalization()(model)
        model = Dropout(rate=0.2, input_shape=(200,))(model)
        model = Dense(100, activation='elu')(model)
        model = Dropout(rate=0.2, input_shape=(100,))(model)
        model = BatchNormalization()(model)
        model = Dense(50, activation='elu')(model)
        model = BatchNormalization()(model)
        model = Dense(20, activation='elu')(model)
        model = BatchNormalization()(model)
        model = Dense(10, activation='elu')(model)
        model = BatchNormalization()(model)
        if self.prediction_type == 'policy':
            activation = 'softmax'
            loss = 'categorical_crossentropy'
            output_dim = 4
        elif self.prediction_type == 'state_value':
            activation = 'sigmoid'
            loss = 'binary_crossentropy'
            output_dim = 1
        model = Dense(output_dim, activation=activation)(model)

        m = Model(inputs, model)
        m.compile(optimizer='adam',
                  loss=loss,
                  metrics=['accuracy', 'AUC'])
        self.model = m
    
    def get_weights(self):
        pass
    
    def replace_weights(self, weights):
        pass
    
    def freeze_layers(self):
        pass

    def load(self, name):
        path = f'../models/{name}'
        self.model = keras.models.load_model(path)
        return self.model


    def save(self, name):
        path = f'../models/{name}'
        self.model.save(path)
def reset_weights(model):
    session = K.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)
        if hasattr(layer, 'bias_initializer'):
            layer.bias.initializer.run(session=session)

class perf_callback(Callback):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.X)
        perf = roc_auc_score(self.y, y_pred, average='micro')
        if self.y.shape[1] > 1:
            accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(self.y, axis=1))
        elif self.y.shape[1] == 1:
            accuracy = np.nan#np.mean((pred.reshape(-1) >=0.5) == (self.y.reshape(-1) == 1))
            # Can't figure out what predictions for state-value model is not same dimension as target
        print(f'Validation auc: {perf}')
        print(f'Validation accuracy: {accuracy}')
        logs['validation'] = perf
        

early_stop = EarlyStopping(patience=1,
                           monitor='validation',
                           mode='max')
tf.compat.v1.reset_default_graph()

# policy_model = nnModel('policy')
# policy_model.build_model()

v_model = nnModel('state_value')
v_model.load('rule_based_v')
nb_passes = 20
nb_files = 41

for it in range(nb_passes):
    print(f'Pass #{it}')
    for i in range(nb_files):
        path = f'../data/it_{i}.pkl'
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print(f'learning using "{path}"')
        X = data['state']
        y = data['y']
        v = data['v']
        X_train, X_test, y_train, y_test, v_train, v_test = train_test_splitter(X, y, 0.05, v=v)
        v_callbacks = [perf_callback(X_test, v_test),
                        early_stop]
        policy_callbacks = [perf_callback(X_test, y_test),
                            early_stop]
        
        print("v model")
        v_model.model.fit(X_train,
                           v_train,
                           epochs=1,
                           batch_size=4096,
                           callbacks=v_callbacks)
        v_model.save('rule_based_v')
    v_model.save('rule_based_v')
        
#         print("Policy model")
#         policy_model.model.fit(X_train,
#                                y_train,
#                                epochs=1,
#                                batch_size=32,
#                                callbacks=policy_callbacks)
        
v_model.save('rule_based_v')
