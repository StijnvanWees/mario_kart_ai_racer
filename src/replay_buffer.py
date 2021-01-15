import tensorflow as tf
import numpy as np
from timeit import default_timer

from src.config import *
from src.db_registry import registry
from src.detection_models import detection_models
from src.shared_process_data import shared_process_data
from src.screen_recorder import make_screen_shot


REPLAY_BUFFER_ELEMENT_SPEC = \
            (tf.TensorSpec((135, 180, 3), dtype=tf.float16),
             (tf.TensorSpec((3,), dtype=tf.int32),
              tf.TensorSpec((3,), dtype=tf.int32)))


def storage_process(q):
    screenshots = q.get()
    actions_0 = q.get()
    actions_1 = q.get()
    game_info = q.get()

    with tf.device("cpu:0"):
        dataset = tf.data.Dataset.from_tensor_slices(
            (screenshots, (actions_0, actions_1))
        )

        tf.data.experimental.save(dataset, REPLAY_BUFFER_PATH + game_info['race_id'])
    registry.add_race(game_info)


class ReplayBuffer:
    def __init__(self):
        pass

    def store_race_and_return_game_info(self, game_info, screenshots, actions):
        game_info["frame_count"] = len(screenshots)

        if ULTIMATE_PERFORMANCE_MODE:
            detection_models.detect_position(make_screen_shot((360, 270)))
            game_info["finished_position"] = detection_models.get_current_detected_position()
            registry.add_race(game_info)
            return game_info

        actions_t = np.array(actions)

        from_frame_index = SKIP_BEGIN_FRAMES
        to_frame_index = game_info["frame_count"] - 1 - SKIP_END_FRAMES
        game_info["frame_count"] = to_frame_index - from_frame_index

        game_info["finished_position"] = detection_models.get_current_detected_position()

        shared_process_data.get_q().put(screenshots[from_frame_index:to_frame_index])
        shared_process_data.get_q().put(list(actions_t[:, 0])[from_frame_index:to_frame_index])
        shared_process_data.get_q().put(list(actions_t[:, 1])[from_frame_index:to_frame_index])
        shared_process_data.get_q().put(game_info)

        process = shared_process_data.get_mp().Process(target=storage_process, args=(shared_process_data.get_q(),))
        shared_process_data.set_process(process)
        #shared_process_data.get_process().start()

        return game_info


replay_buffer = ReplayBuffer()
