from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col

import numpy as np
import pickle
def geese_heads(obs_dict, config_dict):
    """
    Return the position of the geese's heads
    """
    configuration = Configuration(config_dict)

    observation = Observation(obs_dict)
    player_index = observation.index
    player_goose = observation.geese[player_index]
    player_head = player_goose[0]
    player_row, player_column = row_col(player_head, configuration.columns)
    positions = []
    for geese in observation.geese:
        if len(geese)>0:
            geese_head = geese[0]
            row, column = row_col(geese_head, configuration.columns)
        else:
            row = None
            column = None
        positions.append((row, column))
    return positions

def get_last_actions(previous_geese_heads, heads_positions):

    def get_last_action(prev, cur):
        last_action = None

        prev_row = prev[0]
        prev_col = prev[1]
        cur_row = cur[0]
        cur_col = cur[1]

        if cur_row is not None:
            if (cur_row-prev_row == 1) | ((cur_row==0) & (prev_row==6)):
                last_action = Action.SOUTH.name
            elif (cur_row-prev_row == -1) | ((cur_row==6) & (prev_row==0)):
                last_action = Action.NORTH.name
            elif (cur_col-prev_col == 1) | ((cur_col==0) & (prev_col==10)):
                last_action = Action.EAST.name
            elif (cur_col-prev_col == -1) | ((cur_col==10) & (prev_col==0)):
                last_action = Action.WEST.name

        return last_action

    if len(previous_geese_heads) == 0:
        actions = [Action.SOUTH.name, Action.NORTH.name, Action.EAST.name, Action.WEST.name]
        nb_geeses = len(heads_positions)
        last_actions = ["None" for _ in range(nb_geeses)]
    else:   
        last_actions = [get_last_action(*pos) for pos in zip(previous_geese_heads, heads_positions)]

    return last_actions
    
def central_state_space(obs_dict, config_dict, last_actions):
    """
    Recreating a board where my agent's head in the middle of the board 
    (position (4,5)), and creating features accordingly
    """
    
    last_actions_dict = {
        Action.WEST.name: 1,
        Action.EAST.name: 2,
        Action.NORTH.name: 3,
        Action.SOUTH.name: 4,
        "None": 16
    }
    
    configuration = Configuration(config_dict)

    observation = Observation(obs_dict)
    player_index = observation.index
    player_goose = observation.geese[player_index]
    player_head = player_goose[0]
    longuest_opponent = 0
    for i, goose in enumerate(observation.geese):
        if i != player_index:
            opponent_length = len(goose)
            if opponent_length > longuest_opponent:
                longuest_opponent = opponent_length
    player_row, player_column = row_col(player_head, configuration.columns)
    row_offset = player_row - 3
    column_offset = player_row - 5

    foods = observation['food']

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

    food1_row, food1_column = centralize(*row_col(foods[0], configuration.columns))
    food2_row, food2_column = centralize(*row_col(foods[1], configuration.columns))

    food1_row_feat = float(food1_row - 3)/5 if food1_row>=3 else float(food1_row - 3)/5
    food2_row_feat = float(food2_row - 3)/5 if food2_row>=3 else float(food2_row - 3)/5

    food1_col_feat = float(food1_column - 5)/5 if food1_column>=5 else float(food1_column - 5)/5
    food2_col_feat = float(food2_column - 5)/5 if food2_column>=5 else float(food2_column - 5)/5

    # Create the grid
    board = np.zeros([7, 11])
    # Add food to board
    board[food1_row, food1_column] = 15
    board[food2_row, food2_column] = 15

    for geese_id, geese in enumerate(observation.geese):
        nb_blocks = len(geese)
        if nb_blocks > 0:
            for i, pix in enumerate(geese[::-1]):
                if ((i+1) == nb_blocks): #This is the head
                    idx = last_actions_dict[last_actions[geese_id]] #head
                else:
                    idx = (i+5) if (i+5)<15 else 14
                row, col = centralize(*row_col(pix, configuration.columns))
                board[row, col] = idx
            
    return board, len(player_goose), longuest_opponent, food1_row_feat, food1_col_feat, food2_row_feat, food2_col_feat
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
        
        board, player_goose_len, longuest_opponent, food1_row_feat, food1_col_feat, food2_row_feat, food2_col_feat = central_state_space(obs_dict, config_dict, last_actions)
        
        cur_obs = {}
        cur_obs['food1_col'] = food1_col_feat
        cur_obs['food2_col'] = food2_col_feat
        cur_obs['food1_row'] = food1_row_feat
        cur_obs['food2_row'] = food2_row_feat
        cur_obs['goose_size'] = player_goose_len
        cur_obs['longuest_opponent'] = longuest_opponent
        cur_obs['board'] = board
        cur_obs['hunger'] = -1 + (float(obs_dict['step']%40)/20)
        cur_obs['step'] = (float(obs_dict['step'])/100) - 1
        
        return cur_obs, heads_positions, last_actions
    def __call__(self, obs_dict, config_dict):
        cur_obs, heads_positions, last_actions = self.getStateSpace(obs_dict, config_dict)
        
        food1_col_feat = cur_obs['food1_col'] 
        food2_col_feat = cur_obs['food2_col'] 
        food1_row_feat = cur_obs['food1_row']
        food2_row_feat = cur_obs['food2_row']
        player_goose_len = cur_obs['goose_size']
        board = cur_obs['board'] 
        cur_obs['hunger'] = -1 + (float(obs_dict['step']%40)/20)
        cur_obs['step'] = (float(obs_dict['step'])/100) - 1

        self.stateSpace = cur_obs
        
        # Prioritize food that is closer
        if (abs(food1_row_feat) + abs(food1_col_feat)) <= (abs(food2_row_feat) + abs(food2_col_feat)):
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
            if (40 - obs_dict['step']%40) < 6:
                hunger_boost = 10
            elif (40 - obs_dict['step']%40) < 3:
                hunger_boost = 100
                
        action_dict[Action.WEST.name] = 0
        # Is action eligible?
        if (self.last_action is None) | (self.last_action != Action.EAST.name):
            action_dict[Action.WEST.name] += 1E7
        # Will the action kill us right away?
        if (board[3, 4] == 0) | (board[3, 4] == 15):
            action_dict[Action.WEST.name] += 1E6
        # Will the action kill us on the subsequent step?:
        if not((board[2, 4] in list(range(6,15))) & (board[3, 3] in list(range(6,15))) & (board[4, 4] in list(range(6,15)))):
            action_dict[Action.WEST.name] += 1E5
        # Could the action kill us on the subsequent step? - is there a head nearby?
        if not((board[2, 4] in list(range(1,5))) | (board[3, 3] in list(range(1,5))) | (board[4, 4] in list(range(1,5)))):
            action_dict[Action.WEST.name] += 1E4
        # Could the action kill us on the subsequent step? - is there a head further?
        if  (not ((board[2, 3] in list(range(1,5))) | (board[3, 2] in list(range(1,5))) | (board[4, 3] in list(range(1,5))))):
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
        if not((board[2, 6] in list(range(6,15))) & (board[3, 7] in list(range(6,15))) & (board[4, 6] in list(range(6,15)))):
            action_dict[Action.EAST.name] += 1E5
        # Could the action kill us on the subsequent step? - is there a head nearby?
        if not((board[2, 6] in list(range(1,5))) | (board[3, 7] in list(range(1,5))) | (board[4, 6] in list(range(1,5)))):
            action_dict[Action.EAST.name] += 1E4
        # Could the action kill us on the subsequent step? - is there a head further?
        if  (not ((board[2, 7] in list(range(1,5))) | (board[3, 8] in list(range(1,5))) | (board[4, 7] in list(range(1,5))))):
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
        if not((board[2, 4] in list(range(6,15))) & (board[2, 6] in list(range(6,15))) & (board[1, 5] in list(range(6,15)))):
            action_dict[Action.NORTH.name] += 1E5
        # Will the action kill us on the subsequent step?  - is there a head nearby?
        if not((board[2, 4] in list(range(1,5))) | (board[2, 6] in list(range(1,5))) | (board[1, 5] in list(range(1,5)))):
            action_dict[Action.NORTH.name] += 1E4
        # Could the action kill us on the subsequent step? - is there a head further?
        if  (not ((board[1, 4] in list(range(1,5))) | (board[0, 5] in list(range(1,5))) | (board[1, 6] in list(range(1,5))))):
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
        if not((board[4, 4] in list(range(6,15))) & (board[4, 6] in list(range(6,15))) & (board[5, 5] in list(range(6,15)))):
            action_dict[Action.SOUTH.name] += 1E5
        # Will the action kill us on the subsequent step? - is there a head nearby?
        if not((board[4, 4] in list(range(1,5))) | (board[4, 6] in list(range(1,5))) | (board[5, 5] in list(range(1,5)))):
            action_dict[Action.SOUTH.name] += 1E4
        # Could the action kill us on the subsequent step? - is there a head further?
        if  (not ((board[5, 4] in list(range(1,5))) | (board[6, 5] in list(range(1,5))) | (board[5, 6] in list(range(1,5))))):
            action_dict[Action.SOUTH.name] += 1E3
        if (board[5, 5] in [0, 4, 15]) & (board[4, 4] in [0, 1, 15]) & (board[4, 6] in [0, 2, 15]):
            action_dict[Action.SOUTH.name] += 1E2
        if p1_food_row_feat > 0:
            action_dict[Action.SOUTH.name] += 1E1 * hunger_boost
        if p2_food_row_feat > 0:
            action_dict[Action.SOUTH.name] += 1E0 * hunger_boost
            
        west = np.sum(np.isin(board[1:6, 3:6], [0,15]))
        east = np.sum(np.isin(board[1:6, 5:8], [0,15]))
        north = np.sum(np.isin(board[1:4, 3:8], [0,15]))
        south = np.sum(np.isin(board[3:6, 3:8], [0,15]))
        
        pop = np.array([south, north, east, west])
        min_pop = np.argmax(pop)
        
        actions = [Action.SOUTH.name, Action.NORTH.name, Action.EAST.name, Action.WEST.name]
        
        action_dict[actions[min_pop]] += 1000
        
        values = np.array([action_dict[action] for action in actions])
        
        max_equality = (values == max(values))
        possible_actions = (values*max_equality)
        action_pick = np.random.multinomial(1, possible_actions/sum(possible_actions))
        action = actions[np.argmax(action_pick)]

        
        self.last_action = action
        self.last_heads_positions = heads_positions
        return action
rb = RuleBasedAgent()
def agent(obs_dict, config_dict):
    return rb(obs_dict, config_dict)
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
nb_passes = 5
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
                           epochs=20,
                           batch_size=1024,
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
