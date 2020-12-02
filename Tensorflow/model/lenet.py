import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, Activation
from tensorflow.keras.layers import AveragePooling2D, Dense, Flatten


class LeNet(Model):
    def __init__(self):
        super().__init__()

        # Layer names are identical to paper
        self.layerC1 = Sequential(
            Conv2D(filters=6, kernel_size=(5, 5)),
            Activation('relu')
        )

        self.layerS2 = AveragePooling2D(pool_size=(2, 2))

        self.layerC3 = Sequential(
            Conv2D(filters=16, kernel_size=(5, 5)),
            Activation('tanh')
        )

        self.layerS4 = AveragePooling2D(pool_size=(2, 2))

        self.layerC5 = Sequential(
            Conv2D(filters=120, kernel_size=(5, 5)),
            Activation('tanh')
        )

        self.layerF6 = Sequential(
            Flatten(),
            Dense(500),
            Activation('tanh')
        )

        self.outputLayer = Sequential(
            Dense(),
            Activation('softmax')
        )

    def call(self, x):
        x = self.layerC1(x)
        x = self.layerS2(x)
        x = self.layerC3(x)
        x = self.layerS4(x)
        x = self.layerC5(x)
        x = self.layerF6(x)
        x = self.outputLayer(x)

        return x
