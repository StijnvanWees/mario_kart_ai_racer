import tensorflow as tf


class QModel(tf.keras.Model):
    def train_step(self, data):
        x, y, r = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            y_pred = (y_pred[0] * y[0], y_pred[1] * y[1])
            y = (y[0] * r, y[1] * r)

            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


def deep_convolutional_network_v3_():
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

    v_steer = tf.keras.layers.Dense(1, name="v_steer")(l9)
    q_steer = tf.keras.layers.Dense(3, name="steer")(l9)
    v_shoot = tf.keras.layers.Dense(1, name="v_shoot")(l9)
    q_shoot = tf.keras.layers.Dense(3, name="shoot")(l9)

    return QModel(inputs=inputs, outputs=[q_steer, q_shoot])


class DuelingV3Network(tf.keras.Model):
    def __init__(self):
        super(DuelingV3Network, self).__init__()

        self.l3 = tf.keras.layers.Conv2D(64, (5, 5), strides=(3, 3), kernel_initializer="he_normal", use_bias=False)
        self.l3b = tf.keras.layers.BatchNormalization()
        self.l3s = tf.keras.layers.Activation("elu")
        self.l4 = tf.keras.layers.MaxPooling2D((2, 2))
        self.l5 = tf.keras.layers.Conv2D(128, (3, 3), padding="SAME", kernel_initializer="he_normal", use_bias=False)
        self.l5b = tf.keras.layers.BatchNormalization()
        self.l5s = tf.keras.layers.Activation("elu")
        self.l51 = tf.keras.layers.Conv2D(128, (3, 3), padding="SAME", kernel_initializer="he_normal", use_bias=False)
        self.l51b = tf.keras.layers.BatchNormalization()
        self.l51s = tf.keras.layers.Activation("elu")
        self.l6 = tf.keras.layers.MaxPooling2D((2, 2))
        self.l61 = tf.keras.layers.Conv2D(256, (3, 3), padding="SAME", kernel_initializer="he_normal", use_bias=False)
        self.l61b = tf.keras.layers.BatchNormalization()
        self.l62s = tf.keras.layers.Activation("elu")
        self.l63 = tf.keras.layers.MaxPooling2D((2, 2))
        self.l7 = tf.keras.layers.Flatten()
        self.l8 = tf.keras.layers.BatchNormalization()
        self.l81 = tf.keras.layers.Dense(150, kernel_initializer="he_normal", use_bias=False)
        self.l82 = tf.keras.layers.BatchNormalization()
        self.l83 = tf.keras.layers.Activation("elu")
        self.l91 = tf.keras.layers.Dense(100, kernel_initializer="he_normal", use_bias=False)
        self.l92 = tf.keras.layers.BatchNormalization()
        self.l9 = tf.keras.layers.Activation("elu")

        self.v_steer = tf.keras.layers.Dense(1, name="v_steer")
        self.a_steer = tf.keras.layers.Dense(3, name="steer")
        self.v_shoot = tf.keras.layers.Dense(1, name="v_shoot")
        self.a_shoot = tf.keras.layers.Dense(3, name="shoot")

    def call(self, x):
        x = self.l3(x)
        x = self.l3b(x)
        x = self.l3s(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l5b(x)
        x = self.l5s(x)
        x = self.l51(x)
        x = self.l51b(x)
        x = self.l51s(x)
        x = self.l6(x)
        x = self.l61(x)
        x = self.l61b(x)
        x = self.l62s(x)
        x = self.l63(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l81(x)
        x = self.l82(x)
        x = self.l83(x)
        x = self.l91(x)
        x = self.l92(x)
        x = self.l9(x)

        v_steer = self.v_steer(x)
        a_steer = self.a_steer(x)
        v_shoot = self.v_shoot(x)
        a_shoot = self.a_shoot(x)

        q_steer = (v_steer + (a_steer - tf.math.reduce_mean(a_steer, axis=1, keepdims=True)))
        q_shoot = (v_shoot + (a_shoot - tf.math.reduce_mean(a_shoot, axis=1, keepdims=True)))

        return q_steer, q_shoot

    def train_step(self, data):
        x, y, r = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            y_pred = (y_pred[0] * y[0], y_pred[1] * y[1])
            y = (y[0] * r, y[1] * r)

            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


def deep_convolutional_network_v3():
    return DuelingV3Network()
