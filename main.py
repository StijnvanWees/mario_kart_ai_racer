import multiprocessing as mp
import tensorflow as tf

from src.emulator_actions import start_mario_kart_race, kill_mario_kart, process_race, restart_race, start_mario_kart

from src.config import *
from src.policies import policy_manager
from src.detection_models import detection_models
from src.shared_process_data import shared_process_data


class Main:
    def __init__(self):
        self.reload_models()

        if len(TEST_Q_MODELS) > 0:
            self.loop_count = len(TEST_Q_MODELS)
        else:
            self.loop_count = 200

        self.at_loop = 0

        for i in range(self.loop_count):
            self.main_loop()

    def main_loop(self):
        self.drive_tournaments()
        self.reload_models()

    def drive_tournaments(self):
        start_mario_kart()

        for loop_id in range(TRAIN_Q_MODEL_AFTER_PLAY_COUNT):
            self.at_loop += 1
            print("loop ", str(self.at_loop))
            policy_try_factor = ((TRAIN_Q_MODEL_AFTER_PLAY_COUNT - 1.0) - loop_id) / \
                                (TRAIN_Q_MODEL_AFTER_PLAY_COUNT - 1.0)
            policy_try_factor **= 1.5
            policy_try_factor *= MAX_POLICY_TRY_FACTOR

            if ULTIMATE_PERFORMANCE_MODE:
                policy_try_factor = 0.0

            policy_manager.set_policy_try_factor(policy_try_factor)

            game_info = start_mario_kart_race()

            for i in range(1):
                game_info["active_q_model_id"] = policy_manager.get_active_q_model_id()
                game_info["court_id"] = i
                game_info["race_id"] = get_new_id()
                game_info["human_controlled"] = USE_HUMAN_PLAYER_POLICY
                game_info["policy_try_factor"] = policy_manager.get_policy_try_factor()
                game_info["ultimate_performance_mode"] = ULTIMATE_PERFORMANCE_MODE

                process_race(game_info)

            if loop_id < TRAIN_Q_MODEL_AFTER_PLAY_COUNT - 1:
                restart_race(game_info)

            if not ULTIMATE_PERFORMANCE_MODE:
                shared_process_data.get_process().start()

        kill_mario_kart()

    def reload_models(self):
        policy_manager.clear_q_model()
        detection_models.clear_models()

        tf.keras.backend.clear_session()

        policy_manager.reload_q_model()
        detection_models.reload_models()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    shared_process_data.set(mp)
    main = Main()
