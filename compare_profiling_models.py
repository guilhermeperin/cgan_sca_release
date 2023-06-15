from prepare_datasets import *
from profiling_and_attack import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam


class CompareProfilingModels:

    def __init__(self, args, models, dir_results):
        self.args = args
        self.models = models
        self.dir_results = dir_results
        self.datasets = PrepareDatasets(self.args)

    def model(self, classes, number_of_samples, learning_rate=0.001):
        input_shape = (number_of_samples)
        input_layer = Input(shape=input_shape, name="input_layer")

        x = Dense(100, kernel_initializer="glorot_normal", activation="elu")(input_layer)
        x = Dense(100, kernel_initializer="glorot_normal", activation="elu")(x)
        x = Dense(100, kernel_initializer="glorot_normal", activation="elu")(x)
        x = Dense(100, kernel_initializer="glorot_normal", activation="elu")(x)

        output_layer = Dense(classes, activation='softmax', name=f'output')(x)

        m_model = Model(input_layer, output_layer, name='mlp_softmax')
        optimizer = Adam(learning_rate=learning_rate)
        m_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        m_model.summary()
        return m_model

    def run_attack(self):
        """ Load weights into generator """
        self.models.generator.load_weights(
            f"{self.dir_results}/generator_{self.datasets.traces_target_dim}_{self.datasets.traces_reference_dim}_epoch_199.h5")

        ge_real_ta, nt_real_ta, pi_real_ta, _ = template_attack(self.datasets)

        model = self.model(self.datasets.dataset_target.classes, self.datasets.features_dim)
        ge_fake, nt_fake, pi_fake, _ = attack(self.datasets, self.models.generator, self.datasets.features_dim, attack_model=model)

        model = self.model(self.datasets.dataset_target.classes, self.datasets.features_dim)
        ge_real, nt_real, pi_real, _ = attack(self.datasets, self.models.generator, self.datasets.features_dim, attack_model=model,
                                              synthetic_traces=False)

        model = self.model(self.datasets.dataset_target.classes, self.datasets.dataset_target.ns)
        ge_real_original, nt_real_original, pi_real_original, _ = attack(self.datasets, self.models.generator,
                                                                         self.datasets.dataset_target.ns,
                                                                         attack_model=model, synthetic_traces=False, original_traces=True)
