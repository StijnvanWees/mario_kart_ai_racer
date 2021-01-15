import numpy as np
import tensorflow as tf

from src.config import *


class DetectionModels:
    def __init__(self):
        self.preprocessing_model = None
        self.goal_detection_model = None
        self.position_reader_model = None

        self.is_goal_state = 0
        self.frame_count_last_position = 0
        self.current_detected_position = 0

    def clear_models(self):
        self.preprocessing_model = None
        self.goal_detection_model = None
        self.position_reader_model = None

    def reload_models(self):
        self.create_single_player_preprocessing_model()
        self.load_goal_detection_model()

    def create_single_player_preprocessing_model(self):
        inp = tf.keras.Input(shape=(270, 360, 3))
        l1 = tf.keras.layers.experimental.preprocessing.Resizing(135, 180)(inp)
        self.preprocessing_model = tf.keras.Model(inputs=inp, outputs=l1)

    def load_goal_detection_model(self):
        self.position_reader_model = tf.keras.models.load_model(POS_DETECTION_MODEL_PATH)
        self.goal_detection_model = tf.keras.models.load_model(GOAL_DETECTION_MODEL_PATH)

    def get_state_data(self, screenshot):
        return self.preprocessing_model(
            np.reshape(screenshot, (1, 270, 360, 3)))

    def reset_for_race(self):
        self.is_goal_state = 0
        self.frame_count_last_position = 0
        self.current_detected_position = 0

    def detect_goal(self, state_data):
        is_goal = int(round(self.goal_detection_model(state_data)[0].numpy()[0]))

        if is_goal:
            self.is_goal_state += 1
            if self.is_goal_state > 1:
                return True
        else:
            self.is_goal_state = 0

        return False

    def detect_position(self, screenshot):
        screenshot = tf.reshape(screenshot, (1, 270, 360, 3))

        detected_position = np.argmax(self.position_reader_model(screenshot)[0])
        if detected_position > 0:
            self.current_detected_position = detected_position
        return self.current_detected_position

    def detect_end_last_positions(self, screenshot):
        detected_position = self.detect_position(screenshot)

        if detected_position == 0:
            return

        if detected_position == 8:
            self.frame_count_last_position += 1
        else:
            self.frame_count_last_position = 0

        if self.frame_count_last_position > ALLOW_LAST_FOR_FRAMES:
            return True

    def get_current_detected_position(self):
        return self.current_detected_position


detection_models = DetectionModels()
