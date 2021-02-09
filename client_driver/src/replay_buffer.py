import tensorflow as tf
import numpy as np
from timeit import default_timer
from time import sleep

from src.config import *
from src.db_registry import registry
from src.detection_models import detection_models
from src.screen_recorder import make_screen_shot


REPLAY_BUFFER_ELEMENT_SPEC = \
            (tf.TensorSpec((135, 180, 3), dtype=tf.float16),
             (tf.TensorSpec((3,), dtype=tf.int32),
              tf.TensorSpec((3,), dtype=tf.int32)))


def store_race_in_replay_buffer(game_info, state_datas, actions):
    game_info["frame_count"] = len(state_datas)

    if game_info["skip_replay_buffer_data_storage"]:
        detection_models.detect_position(make_screen_shot((360, 270)))
        game_info["finished_position"] = detection_models.get_current_detected_position()
        registry.add_race(game_info)
        sleep(8.0)
        return game_info

    actions_t = np.array(actions)

    from_frame_index = SKIP_BEGIN_FRAMES
    to_frame_index = game_info["frame_count"] - 1 - SKIP_END_FRAMES
    game_info["frame_count"] = to_frame_index - from_frame_index
    game_info["finished_position"] = detection_models.get_current_detected_position()

    state_datas_converted = [tf.cast(state_data * 255.0, dtype=tf.uint8) for state_data in state_datas]

    with tf.device("cpu:0"):
        dataset = tf.data.Dataset.from_tensor_slices(
            (state_datas_converted[from_frame_index:to_frame_index],
             (list(actions_t[:, 0])[from_frame_index:to_frame_index],
              list(actions_t[:, 1])[from_frame_index:to_frame_index]))
        )
        print(dataset.element_spec)
        tf.data.experimental.save(dataset, REPLAY_BUFFER_PATH + game_info['race_id'], compression='GZIP')
    registry.add_race(game_info)

    return game_info
