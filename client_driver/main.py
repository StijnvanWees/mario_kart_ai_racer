from src.mario_kart_sequences import *
from src.config import *
from src.policies import policy_manager
from src.detection_models import detection_models
from src.policy_try_factors import update_policy_try_factor
from src.db_registry import registry

import sys
sys.path.append('.')


class Main:
    def __init__(self):
        self.reload_models()

        if len(TEST_Q_MODELS) > 0:
            self.loop_count = len(TEST_Q_MODELS)
        else:
            self.loop_count = 200

        for i in range(self.loop_count):
            self.main_loop()

    def main_loop(self):
        self.drive_tournaments()
        self.reload_models()

    def drive_tournaments(self):
        start_mario_kart()

        for loop_id in range(TRAIN_Q_MODEL_AFTER_PLAY_COUNT):
            print("loop ", str(loop_id))

            game_info = start_mario_kart_race()

            for i in range(4):
                game_info["active_q_model_id"] = policy_manager.get_active_q_model_id()
                game_info["court_id"] = i
                game_info["race_id"] = get_new_id()
                game_info["human_controlled"] = USE_HUMAN_PLAYER_POLICY
                game_info["ultimate_performance_mode"] = ULTIMATE_PERFORMANCE_MODE
                game_info["skip_replay_buffer_data_storage"] = \
                    registry.get_court_settings_object(game_info).get_skip_replay_buffer_storage()
                game_info["shutdown_when_frames_last"] = \
                    registry.get_court_settings_object(game_info).get_shutdown_when_frames_last()

                if ULTIMATE_PERFORMANCE_MODE:
                    game_info["skip_replay_buffer_data_storage"] = 1

                game_info = update_policy_try_factor(game_info, loop_id, TRAIN_Q_MODEL_AFTER_PLAY_COUNT - 1.0)

                process_race(game_info)

                if game_info["has_finished"] == 0:
                    break
            else:
                kill_mario_kart()
                start_mario_kart()

            if loop_id < TRAIN_Q_MODEL_AFTER_PLAY_COUNT - 1:
                restart_race(game_info)

        kill_mario_kart()

    def reload_models(self):
        policy_manager.clear_q_model()
        detection_models.clear_models()

        tf.keras.backend.clear_session()

        policy_manager.reload_q_model()
        detection_models.reload_models()


if __name__ == '__main__':
    main = Main()
