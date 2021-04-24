from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col

import numpy as np
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
        
        actions = [Action.SOUTH.name, Action.NORTH.name, Action.EAST.name, Action.WEST.name]
        values = np.array([action_dict[action] for action in actions])
        
        max_equality = (values == max(values))
        possible_actions = (values*max_equality)
        action_pick = np.random.multinomial(1, possible_actions/sum(possible_actions))
        action = actions[np.argmax(action_pick)]

        
        self.last_action = action
        self.last_heads_positions = heads_positions
        return action
from random import choice
from copy import deepcopy
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, \
                                                                row_col, adjacent_positions, translate, min_distance
class GreedyAgent:
    def __init__(self):
        
        self.last_action = None
        self.observations = []

    def __call__(self, observation: Observation, configuration: Configuration):
        self.configuration = configuration
        
        board = np.zeros(self.configuration.rows*self.configuration.columns)
        board_shape = (self.configuration.rows, self.configuration.columns)
        
        board_heads = deepcopy(board)
        board_bodies = deepcopy(board)
        board_rewards = deepcopy(board)
        
        
        rows, columns = self.configuration.rows, self.configuration.columns

        food = observation.food
        geese = observation.geese
        
        
        opponents = [
            goose
            for index, goose in enumerate(geese)
            if index != observation.index and len(goose) > 0
        ]

        
        opponent_heads = [opponent[0] for opponent in opponents]
        # Don't move adjacent to any heads
        head_adjacent_positions = {
            opponent_head_adjacent
            for opponent_head in opponent_heads
            for opponent_head_adjacent in adjacent_positions(opponent_head, columns, rows)
        }
        
        tail_adjacent_positions ={
            opponent_tail_adjacent
            for opponent in opponents
            for opponent_tail in [opponent[-1]]
            for opponent_tail_adjacent in adjacent_positions(opponent_tail, columns, rows)
        }
        # Don't move into any bodies
        #bodies, heads = [position for goose in geese for position in goose]
        
        heads = [i[0] for i in geese if len(i)>1]
        bodies = [item for sublist in geese for item in sublist]
        
        board_bodies[list(bodies)] = 1
        board_heads[heads] = 1

        # Move to the closest food
        position = geese[observation.index][0]
        actions = {
            action: min_distance(new_position, food, columns)
            for action in Action
            for new_position in [translate(position, action, columns, rows)]
            if (
                new_position not in head_adjacent_positions and
                new_position not in bodies and
                (self.last_action is None or action != self.last_action.opposite())
            )
        }

        action = min(actions, key=actions.get) if any(actions) else choice([action for action in Action])
        
        
        cur_obs = {}
        cur_obs['head_adjacent_positions'] = head_adjacent_positions
        cur_obs['bodies'] = bodies
        cur_obs['board_bodies'] = board_bodies.reshape(board_shape)
        cur_obs['board_heads'] = board_heads.reshape(board_shape)
        cur_obs['tails'] = tail_adjacent_positions
        cur_obs['actions'] = actions
        cur_obs['action'] = action
        cur_obs['last_action'] = self.last_action
#         cur_obs['goose_size'] = player_goose_len
#         cur_obs['board'] = board
        cur_obs['cur_action'] = action
        self.observations.append(cur_obs)
        
        self.last_action = action
        return action.name


cached_greedy_agents = {}


def greedy_agent(obs, config):
    index = obs["index"]
    if index not in cached_greedy_agents:
        cached_greedy_agents[index] = GreedyAgent(Configuration(config))
    return cached_greedy_agents[index](Observation(obs))
actions_list = np.array(['EAST',
                        'WEST',
                        'SOUTH',
                        'NORTH'])
def action_to_target(action):
    pos = np.argmax(actions_list == action)
    target = np.zeros(4)
    target[pos] = 1
    return target

def target_to_action(target):
    pos = np.argmax(target)
    return actions_list[pos]

def pred_to_action(pred):
    pos = np.argmax(np.random.multinomial(1, pred))
    return actions_list[pos]
def add_numerical(steps):
    numerical = ['goose_size',
                 'longuest_opponent',
                 'hunger',
                 'step']
    food_position = ['food1_col',
                     'food2_col',
                     'food1_row',
                     'food2_row']
    for step in steps:
        numerical_vector = np.zeros(len(numerical)+1)
        for i, nm in enumerate(numerical):
            if nm in ['goose_size', 'longuest_opponent']:
                numerical_vector[i] = (float(step['cur_state'][nm])-8)/16
            else:
                numerical_vector[i] = step['cur_state'][nm]
        food_position_vector = np.zeros(len(food_position))
        for i, nm in enumerate(food_position):
            food_position_vector[i] = step['cur_state'][nm]
        numerical_vector[len(numerical)] = float(step['cur_state']['goose_size'] - step['cur_state']['longuest_opponent'])/10
        step['numerical'] = numerical_vector
        step['food_position_vector'] = food_position_vector
    return None
def add_embeddings(steps):
    numerical = ['food1_col',
                'food2_col',
                'food1_row',
                'food2_row',
                'goose_size',
                'hunger',
                'step']
    for step in steps:
        #vector = np.zeros(7*11, dtype=int)
        vector = []
        board = step['cur_state']['board']
        for row in range(7):
            for col in range(11):
               #vector[11*row + col] =  np.array(board[row][col], dtype=int)
               vector.append(int(board[row][col]))
        step['embeddings'] = vector
    return None
def add_state_value(discount, steps):
    steps_back = steps[::-1]
    v_prime = 0
    for step in steps_back:
        v = step['reward'] + discount*v_prime
        v_prime = v
        step['v'] = v
    return None
def add_next_state(steps):
    nb_steps = len(steps)
    for i, step in enumerate(steps):
        if step['status'] == 'ACTIVE':
            step['next_embeddings'] = steps[i+1]['embeddings']
            step['next_food_position_vector'] = steps[i+1]['food_position_vector']
            step['next_numerical'] = steps[i+1]['numerical']
        else:
            step['next_embeddings'] = None
            step['next_food_position_vector'] = None
            step['next_numerical'] = None
def process(discount, episodes):
    for episode in episodes:
        add_embeddings(episode)
        add_numerical(episode)
        add_state_value(discount, episode)
        add_next_state(episode)
    return None
def training_data(episodes):
    targets = []
    next_numerical = []
    next_embeddings = []
    reward = []
    done = []
    v = []
    actions = []
    numerical = []
    embeddings = []
    for episode in episodes:
        for step in episode:
            action = step['action']
            target = action_to_target(action)
            targets.append(target)
            num = step['numerical']
            emb = step['embeddings']
            next_numerical = step['next_numerical']
            next_embeddings = step['next_embeddings']
            numerical.append(num)
            embeddings.append(emb)
            actions.append(action)
            v.append(step['v'])
            done.append(step['done'])
            reward.append(step['reward'])

    target_reshaped = np.array(targets).reshape(-1, 4)
    e = [np.array(embeddings)[:, i].reshape(-1, 1) for i in range(7*11)]
    n = [np.array(numerical)[:, i].reshape(-1, 1) for i in range(5)]
    train = n+e

    if not done:
        e_next = [np.array(next_embeddings)[:, i].reshape(-1, 1) for i in range(7*11)]
        n_next = [np.array(next_numerical)[:, i].reshape(-1, 1) for i in range(5)]
        train_next = n_next+e_next
    else:
        train_next = None

    training_dict = {'state': train,
                     'action': action,
                     'next_state': train_next,
                     'y': target_reshaped,
                     'reward': reward,
                     'v': v,
                     'done': done}
    return training_dict
step_reward = 0
dying_reward = 0
winning_reward = 1
step_200_reward = lambda my_goose, longuest_opponent: winning_reward if my_goose > longuest_opponent else 0
win_game_reward = lambda step, my_goose, longuest_opponent: winning_reward #max((200-step), winning_reward)

discount = 1

nb_opponents = 1

steps_per_ep = 200
num_episodes = 1000


env = make("hungry_geese", debug=True)
config = env.configuration
import pickle
for it in range(100):
    print(f'starting iteration {it}')
    name = f'it_{it}.pkl'
    episodes = []
    for ep in range(num_episodes):
        print('episode number: ', ep)
        steps = []
        my_agent = RuleBasedAgent()
        agents =  [my_agent] + [(RuleBasedAgent() if np.random.rand()<0.7 else GreedyAgent()) for _ in range(nb_opponents)]
        state_dict = env.reset(num_agents=nb_opponents + 1)[0]
        observation = state_dict['observation']
        my_goose_ind = observation['index']

        reward = state_dict['reward']
        action = state_dict['action']



        done = False
        for step in range(1, steps_per_ep):
            actions = []

            for i, agent in enumerate(agents):
                obs = deepcopy(observation)
                obs['index'] = i
                action = agent(obs, config)
                actions.append(action)

            state_dict = env.step(actions)[0]
            observation = state_dict['observation']
            my_goose_ind = observation['index']

            my_goose_length = len(observation['geese'][my_goose_ind])

            longuest_opponent=0
            for i, goose in enumerate(observation.geese):
                if i != my_goose_ind:
                    opponent_length = len(goose)
                    if opponent_length > longuest_opponent:
                        longuest_opponent = opponent_length

            #new_state, _, _ = agent.getStateSpace(observation, config)

            #reward = state_dict['reward']
            action = state_dict['action']
            status = state_dict['status']

            if status != "ACTIVE":
                done = True

            # Check if my goose died
            if my_goose_length == 0:
                done = True
                reward = dying_reward
            elif (step+1) == steps_per_ep:
                reward = step_200_reward(my_goose_length, longuest_opponent)
                done = True
            elif status != "ACTIVE":
                reward = win_game_reward(step, my_goose_length, longuest_opponent)
            else:
                reward = step_reward

            steps.append({'cur_state': my_agent.stateSpace,
                                    'action': action,
                                    'reward': reward,
                                    'new_state': '',#new_state,
                                    'status': status,
                                    'done': done})
            if done:
#                 print('Done, Step: ', step+1)
#                 print('status, ', status)
                break

            if step%50 == 0:
                pass
                #print(f'We survived {step+1} steps')
        episodes.append(steps)
    process(discount, episodes)
    train_data = training_data(episodes)
    with open(f'../data/{name}', 'wb') as f:
        pickle.dump(train_data, f)
