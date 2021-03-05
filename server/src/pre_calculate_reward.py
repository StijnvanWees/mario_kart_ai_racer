import tensorflow as tf
from os import listdir

from src.db_registry import registry
from src.config import *


REPLAY_BUFFER_MERGED_ELEMENT_SPEC =  \
            (tf.TensorSpec((135, 180, 3), dtype=tf.uint8),
             (tf.TensorSpec((3,), dtype=tf.uint8),
              tf.TensorSpec((3,), dtype=tf.uint8)))


class PreCalculateReward:
    def __init__(self):
        self.reward_detection_model = tf.keras.models.load_model(REWARD_DETECTION_MODEL_PATH)

        self.pre_calculate()
        self.free_system_resources()

    def pre_calculate(self):
        replay_filenames = [obj.get_id() for obj in registry.get_race_data_objects_in_replay_buffer()]
        reward_filenames = [obj for obj in listdir(REWARD_BUFFER_PATH)]

        if RECALCULATE_REWARD:
            filenames = replay_filenames
        else:
            filenames = [obj for obj in replay_filenames
                         if obj not in reward_filenames]

        for i, filename in enumerate(filenames):
            self.batch_calculations_dataset(filename)

            if (i + 1) % 10 == 0:
                print(f"pre calculated reward for {i + 1} from {len(filenames)} races")

        print(f"Completed reward pre calculation for {len(filenames)} races")

    def batch_calculations_dataset(self, filename):
        dataset = tf.data.experimental.load(REPLAY_BUFFER_PATH + filename,
                                                 REPLAY_BUFFER_MERGED_ELEMENT_SPEC, compression="GZIP")
        dataset = dataset.batch(BATCH_SIZE_REWARD_PRE_CALCULATION).prefetch(tf.data.experimental.AUTOTUNE)

        reward_batches = []

        for elem in dataset:
            reward_params = self.reward_detection_model(tf.cast(elem[0], dtype=tf.float32)  / 255.0)

            msbs = tf.math.argmax(reward_params[:][0], axis=1)
            lsbs = tf.math.argmax(reward_params[:][1], axis=1)
            poss = tf.math.argmax(reward_params[:][2], axis=1)
            revs = tf.reshape(tf.math.round(reward_params[:][3]), [-1])

            speeds = tf.cast(msbs * 10 + lsbs, dtype=tf.float32)

            position_factor = ((tf.constant(11.0, dtype=tf.float32) - tf.cast(poss, dtype=tf.float32)) / 10.0)

            rewards = speeds * position_factor * (tf.constant(1.0, dtype=tf.float32) - revs)

            reward_batches.append(rewards)

        reward = tf.concat(reward_batches, 0)
        dataset = tf.data.Dataset.from_tensor_slices(reward)
        tf.data.experimental.save(dataset, REWARD_BUFFER_PATH + filename)

    def free_system_resources(self):
        self.reward_detection_model = None
        tf.keras.backend.clear_session()
