from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.initializers import RandomNormal
import tensorflow as tf

class CharCNN(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, max_length, num_classes, feature='small', padding='same', trainable=True, name=None, dtype=None,input_shape=None):
        super(CharCNN, self).__init__(name=name, dtype=dtype)
        assert feature in ['small', 'large'], "Feature must be either 'small' or 'large'"
        assert padding in ['valid', 'same'], "Padding must be either 'valid' or 'same'"

        #assigning values passed as arguments to instance variables of the CharCNN object
        self.padding = padding
        self.num_classes = num_classes
        self.trainable = trainable

        # Configurations based on model size
        if feature == 'small':
            self.units_fc = 1024  # Units in fully connected layers
            self.num_filter = 256
            self.stddev = 0.05
        else:
            self.units_fc = 2048
            self.num_filter = 1024
            self.stddev = 0.02

        # Weight initialization for convolutional layers
        self.initializers = RandomNormal(mean=0., stddev=self.stddev, seed=42)

        # Model configuration
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.max_length = max_length

        # Embedding layer
        self.embedding = Embedding(self.vocab_size, self.embedding_size, input_length=self.max_length, trainable=self.trainable)

        # Convolutional blocks
        self.conv1d_1 = Conv1D(self.num_filter, kernel_size=7, kernel_initializer=self.initializers, activation='relu', padding=self.padding, trainable=self.trainable)
        self.maxpooling1d_1 = MaxPooling1D(pool_size=3)

        self.conv1d_2 = Conv1D(self.num_filter, kernel_size=7, kernel_initializer=self.initializers, activation='relu', padding=self.padding, trainable=self.trainable)
        self.maxpooling1d_2 = MaxPooling1D(pool_size=3)

        self.conv1d_3 = Conv1D(self.num_filter, kernel_size=3, kernel_initializer=self.initializers, activation='relu', padding=self.padding, trainable=self.trainable)
        self.conv1d_4 = Conv1D(self.num_filter, kernel_size=3, kernel_initializer=self.initializers, activation='relu', padding=self.padding, trainable=self.trainable)
        self.conv1d_5 = Conv1D(self.num_filter, kernel_size=3, kernel_initializer=self.initializers, activation='relu', padding=self.padding, trainable=self.trainable)

        self.conv1d_6 = Conv1D(self.num_filter, kernel_size=3, kernel_initializer=self.initializers, activation='relu', padding=self.padding, trainable=self.trainable)
        self.maxpooling1d_6 = MaxPooling1D(pool_size=3)

        # Fully connected layers
        self.flatten = Flatten()
        self.fc1 = Dense(self.units_fc, activation='relu', trainable=self.trainable)
        self.drp1 = Dropout(0.5)
        self.fc2 = Dense(self.units_fc, activation='relu', trainable=self.trainable)
        self.drp2 = Dropout(0.5)
        self.fc3 = Dense(self.num_classes, activation='softmax', trainable=self.trainable)

    def call(self, inputs):
        # Forward pass
        x = self.embedding(inputs)
        x = self.maxpooling1d_1(self.conv1d_1(x))
        x = self.maxpooling1d_2(self.conv1d_2(x))
        x = self.conv1d_3(x)
        x = self.conv1d_4(x)
        x = self.conv1d_5(x)
        x = self.maxpooling1d_6(self.conv1d_6(x))

        x = self.flatten(x)

        # Fully connected layers
        x = self.drp1(self.fc1(x))
        x = self.drp2(self.fc2(x))
        outputs = self.fc3(x)

        return outputs

    def get_config(self):
        """Returns the config of the model for serialization."""
        config = super(CharCNN, self).get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embedding_size": self.embedding_size,
            "max_length": self.max_length,
            "num_classes": self.num_classes,
            "feature": "small" if self.units_fc == 1024 else "large",
            "padding": self.padding,
            "trainable": self.trainable
        })
        return config

    # @classmethod
    # def from_config(cls, config):
    #     """Creates a model instance from its config."""
    #     return cls(**config)

    @classmethod
    def from_config(cls, config):
      config = config.copy()
      dtype = config.pop('dtype', None)
      return cls(dtype=dtype, **config)
