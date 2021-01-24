import tensorflow as tf
from PIL import ImageGrab, Image


def make_screen_shot(size):
    return tf.keras.preprocessing.image.img_to_array(
        ImageGrab.grab(bbox=(240, 0, 1680, 1080)).resize(size, resample=Image.BILINEAR), dtype="float16") / 255.0
