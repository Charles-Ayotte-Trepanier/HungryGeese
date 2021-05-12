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
    pos = np.argmax(np.random.multinomial(1, pred[0]))
    return actions_list[pos]


def pred_to_action_greedy(pred):
    pos = np.argmax(pred)
    return actions_list[pos]
