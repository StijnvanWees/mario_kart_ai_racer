import tensorflow as tf


class QModel(tf.keras.Model):
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)

            # Mask Q-values of non performed actions:
            y_pred = list([y_pred[i] * tf.linalg.normalize(y[i], axis=1)[0] for i in range(len(y_pred))])

            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


def deep_convolutional_network_v3():
    inputs = tf.keras.Input(shape=(135, 180, 3))

    l3 = tf.keras.layers.Conv2D(64, (5, 5), strides=(3, 3), kernel_initializer="he_normal", use_bias=False)(inputs)
    l3b = tf.keras.layers.BatchNormalization()(l3)
    l3s = tf.keras.layers.Activation("elu")(l3b)
    l4 = tf.keras.layers.MaxPooling2D((2, 2))(l3s)
    l5 = tf.keras.layers.Conv2D(128, (3, 3), padding="SAME", kernel_initializer="he_normal", use_bias=False)(l4)
    l5b = tf.keras.layers.BatchNormalization()(l5)
    l5s = tf.keras.layers.Activation("elu")(l5b)
    l51 = tf.keras.layers.Conv2D(128, (3, 3), padding="SAME", kernel_initializer="he_normal", use_bias=False)(l5s)
    l51b = tf.keras.layers.BatchNormalization()(l51)
    l51s = tf.keras.layers.Activation("elu")(l51b)
    l6 = tf.keras.layers.MaxPooling2D((2, 2))(l51s)
    l61 = tf.keras.layers.Conv2D(256, (3, 3), padding="SAME", kernel_initializer="he_normal", use_bias=False)(l6)
    l61b = tf.keras.layers.BatchNormalization()(l61)
    l62s = tf.keras.layers.Activation("elu")(l61b)
    l63 = tf.keras.layers.MaxPooling2D((2, 2))(l62s)
    l7 = tf.keras.layers.Flatten()(l63)
    l8 = tf.keras.layers.BatchNormalization()(l7)
    l81 = tf.keras.layers.Dense(150, kernel_initializer="he_normal", use_bias=False)(l8)
    l82 = tf.keras.layers.BatchNormalization()(l81)
    l83 = tf.keras.layers.Activation("elu")(l82)
    l91 = tf.keras.layers.Dense(100, kernel_initializer="he_normal", use_bias=False)(l83)
    l92 = tf.keras.layers.BatchNormalization()(l91)
    l9 = tf.keras.layers.Activation("elu")(l92)

    out_steer = tf.keras.layers.Dense(3, name="steer", activation="relu")(l9)
    out_shoot = tf.keras.layers.Dense(3, name="shoot", activation="relu")(l9)

    return QModel(inputs=inputs, outputs=[out_steer, out_shoot])
