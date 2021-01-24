import tensorflow as tf
import numpy as np

import keyboard
from random import randint
from os import listdir

from src.config import *
from src.db_registry import registry


class HumanPlayerPolicy:
    def __init__(self):
        self.shot = [1, 0, 0]
        self.steer = [0, 1, 0]

        keyboard.on_press_key("c", self.shoot)
        keyboard.on_press_key("left", self.left)
        keyboard.on_press_key("right", self.right)

    def shoot(self, k_event):
        if self.shot[0] == 0:
            return

        if keyboard.is_pressed('down'):
            self.shot = [0, 1, 0]
        else:
            self.shot = [0, 0, 1]

    def left(self, k_event):
        self.steer = [1, 0, 0]

    def right(self, k_event):
        self.steer = [0, 0, 1]

    def next_action(self):
        if keyboard.is_pressed('left'):
            act_0 = np.array([1, 0, 0], dtype=np.int32)
        elif keyboard.is_pressed('right'):
            act_0 = np.array([0, 0, 1], dtype=np.int32)
        else:
            act_0 = np.array(self.steer, dtype=np.int32)

        action = np.array((act_0, np.array(self.shot, dtype=np.int32)))
        self.shot = [1, 0, 0]
        self.steer = [0, 1, 0]
        return action


class QModelPolicy:
    def __init__(self):
        self.active_q_model_id = ""
        self.q_model = None

    def clear_q_model(self):
        self.q_model = None
        #self.active_q_model_id = ""

    def reload_q_model(self):
        if TEST_Q_MODELS:
            if not self.active_q_model_id:
                self.active_q_model_id = TEST_Q_MODELS[0]
            else:
                at_index = 0
                for test_q_model in TEST_Q_MODELS:
                    at_index += 1
                    if test_q_model == self.active_q_model_id:
                        break
                self.active_q_model_id = TEST_Q_MODELS[min(at_index, len(TEST_Q_MODELS) - 1)]
        else:
            q_model_id = registry.get_latest_q_model_object().get_id()
            if q_model_id[-1] == "X":
                filenames = [filename for filename in listdir(Q_MODEL_PATH)
                             if filename[:len(q_model_id[:-1])] == q_model_id[:-1]]

                q_model_id = max(filenames)


            self.active_q_model_id = q_model_id

        print("loaded q-model: ", self.active_q_model_id)
        self.q_model = tf.keras.models.load_model(Q_MODEL_PATH + self.active_q_model_id)

    def hardmax_list(self, max_index, length):
        lst = []
        for i in range(length):
            if i == max_index:
                lst.append(1)
            else:
                lst.append(0)
        return np.array(lst, dtype=np.int32)

    def prediction_to_action(self, action_prediction):
        return np.array((self.hardmax_list(np.argmax(action_prediction[0]), 3),
                         self.hardmax_list(np.argmax(action_prediction[1]), 3)))

    def get_q_model(self):
        return self.q_model

    def get_active_q_model_id(self):
        return self.active_q_model_id

    def next_action(self, state_data):
        action_prediction = self.q_model(np.reshape(np.array(state_data), (1, 135, 180, 3)))
        return self.prediction_to_action(action_prediction)


class RandomPolicy:
    def __init__(self):
        self.try_factor = 0.0

    def set_try_factor(self, factor):
        self.try_factor = factor

    def next_action(self):
        random_num = randint(0, 3)
        if random_num == 0:
            return np.array([1, 0, 0], dtype=np.int32)
        elif random_num == 1:
            return np.array([0, 0, 1], dtype=np.int32)
        else:
            return np.array([0, 1, 0], dtype=np.int32)


class PolicyManager:
    def __init__(self):
        self.policy_try_factor = 0.0

        self.human_player_policy = HumanPlayerPolicy()
        self.q_model_policy = QModelPolicy()
        self.random_policy = RandomPolicy()

    def set_policy_try_factor(self, factor):
        self.policy_try_factor = factor

    def get_policy_try_factor(self):
        return self.policy_try_factor

    def get_active_q_model_id(self):
        return self.q_model_policy.get_active_q_model_id()

    def clear_q_model(self):
        self.q_model_policy.clear_q_model()

    def reload_q_model(self):
        self.q_model_policy.reload_q_model()

    def next_action(self, state_data):
        if USE_HUMAN_PLAYER_POLICY:
            return self.human_player_policy.next_action()
        else:
            policy_action = self.q_model_policy.next_action(state_data)
            policy_action = np.array((policy_action[0], policy_action[1]))

            if self.policy_try_factor > (randint(0, 1000) / 1000.0):
                return np.array((self.random_policy.next_action(), policy_action[1]))
            else:
                return policy_action


policy_manager = PolicyManager()
