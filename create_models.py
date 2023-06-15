import tensorflow as tf
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import *
import tensorflow.keras.backend as K
import random


class CreateModels:

    def __init__(self, args, random_hp=False):
        # helper functions for tensorflow compiling
        self.real_accuracy_metric = BinaryAccuracy()
        self.fake_accuracy_metric = BinaryAccuracy()
        self.cross_entropy = BinaryCrossentropy(from_logits=True)
        self.cross_entropy_disc = BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(0.0025, beta_1=0.5)

        if random_hp:
            self.hp_d = {
                "neurons": random.choice([40, 50, 100]),
                "layers": random.choice([3, 4, 5, 6]),
                "activation": random.choice(["elu", "selu", "relu"]),
                "learning_rate": random.choice([0.001, 0.0025, 0.0005, 0.0001, 0.0002, 0.00025, 0.00005]),
                "kernel_initializer": random.choice(
                    ["random_uniform", "he_uniform", "glorot_uniform", "random_normal", "he_normal", "glorot_normal"]),
                "dropout": random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5]),
            }
            self.hp_g = {
                "neurons": random.choice([40, 50, 100]),
                "layers": random.choice([3, 4, 5, 6]),
                "activation": random.choice(["elu", "selu", "relu"]),
                "learning_rate": random.choice([0.001, 0.0025, 0.0005, 0.0001, 0.0002, 0.00025, 0.00005]),
                "kernel_initializer": random.choice(
                    ["random_uniform", "he_uniform", "glorot_uniform", "random_normal", "he_normal", "glorot_normal"]),
                "dropout": random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5]),
            }

        # create the discriminator
        self.discriminator = self.define_discriminator(args["features"],
                                                       n_classes=9 if args["leakage_model"] == "HW" else 256)
        # create the generator
        self.generator = self.define_generator(args["dataset_target_dim"], args["features"])

    def discriminator_loss(self, real, fake):
        real_loss = self.cross_entropy_disc(tf.ones_like(real), real)
        fake_loss = self.cross_entropy_disc(tf.zeros_like(fake), fake)
        return real_loss + fake_loss

    def generator_loss(self, fake):
        return self.cross_entropy(tf.ones_like(fake), fake)

    def define_discriminator(self, features_dim: int, n_classes: int = 256, kern_init='random_normal'):
        # label input
        in_label = Input(shape=1)
        y = Embedding(n_classes, n_classes)(in_label)
        y = Dense(200, kernel_initializer=kern_init)(y)
        # y = LeakyReLU()(y)
        y = Flatten()(y)

        in_features = Input(shape=(features_dim,))

        merge = Concatenate()([y, in_features])

        x = Dense(100, kernel_initializer=kern_init)(merge)
        x = LeakyReLU()(x)
        x = Dropout(0.60)(x)

        # output
        out_layer = Dense(1, activation='sigmoid')(x)

        model = Model([in_label, in_features], out_layer)
        model.summary()
        return model

    def define_discriminator_split_bilinear(self, features_dim: int, n_classes: int = 256, kern_init='random_normal'):
        # label input
        in_label = Input(shape=1)
        l1 = Embedding(n_classes, n_classes)(in_label)
        l1 = Dense(30, kernel_initializer=kern_init)(l1)
        l1 = LeakyReLU()(l1)
        l1 = Flatten()(l1)

        in_features = Input(shape=(features_dim,))

        half_size = features_dim // 2
        first_half = Lambda(lambda x: x[:, :half_size])(in_features)
        second_half = Lambda(lambda x: x[:, half_size:])(in_features)
        dot_lambda = lambda x_arr: tf.multiply(x_arr[0], x_arr[1])
        x = Lambda(dot_lambda)([first_half, second_half])

        # Compute dot product
        # x = Multiply()([first_half, second_half])

        # in_features_reshaped = Reshape((1, features_dim // 2))(in_features[:, features_dim // 2:features_dim])
        # temp = in_features[:, 0:features_dim // 2]
        # dot_lambda = lambda x_arr: tf.multiply(x_arr[0], x_arr[1])
        # in_features_dot = Lambda(dot_lambda)([temp, in_features_reshaped])
        # # Should be (None, 50, 50)
        # print(in_features_dot.shape)
        #
        # x = Flatten()(in_features_dot)
        # # Should be (None, 2500(features_dim^2))
        # print(x.shape)
        x = Dense(30, kernel_initializer=kern_init)(x)
        x = LeakyReLU()(x)

        merge = Concatenate()([l1, x])

        x = Dense(10, kernel_initializer=kern_init)(merge)
        x = LeakyReLU()(x)

        # output
        out_layer = Dense(1, activation='sigmoid')(x)

        # model = Model([input_traces, in_label, in_features], out_layer)
        model = Model([in_label, in_features], out_layer)
        # model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=self.hp_d["learning_rate"]), metrics=['accuracy'])
        model.summary()
        return model

    def define_discriminator_split_bilinear_order(self, features_dim: int, n_classes: int = 256, order=2, kern_init='random_normal'):
        # label input
        in_label = Input(shape=1)
        l1 = Embedding(n_classes, n_classes)(in_label)
        l1 = Dense(30, kernel_initializer=kern_init)(l1)
        l1 = LeakyReLU()(l1)
        l1 = Flatten()(l1)

        in_features = Input(shape=(features_dim, 1))
        temp = in_features[:, 0:features_dim // order]
        for i in range(1, order):
            in_features_reshaped = Reshape((1, features_dim // order))(
                in_features[:, (features_dim * i) // order:(features_dim * (i + 1)) // order])
            # # input_traces = Input(shape=self.traces_dim)
            temp = Reshape((features_dim // order, 1))(temp)
            dot_lambda = lambda x_arr: tf.multiply(x_arr[0], x_arr[1])
            temp = Lambda(dot_lambda)([temp, in_features_reshaped])
            # Should be (None, 50, 50)
            temp = Flatten()(temp)
            temp = Dense(features_dim // order, kernel_initializer=kern_init)(temp)

        # Should be (None, 2500(features_dim^2))
        in_features_dot = LeakyReLU()(temp)
        # merge = Concatenate()([input_traces, l1, in_features])
        merge = Concatenate()([l1, in_features_dot])

        x = Dense(10, kernel_initializer=kern_init)(merge)
        x = LeakyReLU()(x)

        # output
        out_layer = Dense(1, activation='sigmoid')(x)

        # model = Model([input_traces, in_label, in_features], out_layer)
        model = Model([in_label, in_features], out_layer)
        # model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=self.hp_d["learning_rate"]), metrics=['accuracy'])
        # model.summary()
        return model

    # define the standalone generator model
    # def define_generator(self, input_dim: int, output_dim: int, n_classes=256):
    #     # input_random_data = Input(shape=(self.traces_target_dim,))
    #     # rnd = Dense(400, activation='elu')(input_random_data)
    #
    #     in_traces = Input(shape=(input_dim,))
    #     reshaped = Reshape((input_dim, 1))(in_traces)
    #     x = Conv1D(16, 32, kernel_initializer='he_uniform', activation='selu', strides=16, padding='same', name='block1_conv1')(reshaped)
    #     # x = BatchNormalization()(x)
    #     # x = Conv1D(8, 11, kernel_initializer='he_uniform', activation='relu',strides=2,  padding='same', name='block1_conv2')(x)
    #     # x = BatchNormalization()(x)
    #     # x = Conv1D(16, 11, kernel_initializer='he_uniform', activation='relu',strides=2,  padding='same', name='block1_conv3')(x)
    #     # x = BatchNormalization()(x)
    #     x = Flatten(name='flatten')(x)
    #     x = Dense(50, activation='selu')(x)
    #     # x = Dropout(0.2)(x)
    #     out_layer = Dense(output_dim, activation='linear')(x)
    #
    #     model = Model([in_traces], out_layer)
    #     model.summary()
    #     return model

    def add_gaussian_noise(self, x):
        mean = 0.0  # Mean of the Gaussian distribution
        stddev = 0.01  # Standard deviation of the Gaussian distribution
        noise = K.random_normal(shape=K.shape(x), mean=mean, stddev=stddev)
        return x + noise

    # define the standalone generator model
    def define_generator(self, input_dim: int, output_dim: int, n_classes=256):
        # input_random_data = Input(shape=(self.traces_target_dim,))
        # rnd = Dense(400, activation='elu')(input_random_data)

        in_traces = Input(shape=(input_dim,))
        # x = Lambda(self.add_gaussian_noise)(in_traces)
        x = Dense(400, activation='linear')(in_traces)
        x = Dense(200, activation='linear')(x)
        x = Dense(100, activation='linear')(x)
        out_layer = Dense(output_dim, activation='linear')(x)

        model = Model([in_traces], out_layer)
        model.summary()
        return model
