from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization
from keras.optimizers import Adadelta
from keras.optimizers import Adagrad
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D


def nvidia_model():
    # Rectified linear activation function
    activation = 'relu'

    # Create the model used by NVIDIA in 'End-to-End Learning for Self-Driving Cars'
    # With a different image size (320 by 180)
    model = Sequential()
    model.add(BatchNormalization(
        epsilon=0.001,
        axis=1,
        input_shape=(200, 66, 3)
    ))
    model.add(Conv2D(
        filters=24,
        kernel_size=5,
        strides=2,
        activation=activation,
    ))
    model.add(Conv2D(
        filters=36,
        kernel_size=5,
        strides=2,
        activation=activation,
    ))
    model.add(Conv2D(
        filters=48,
        kernel_size=5,
        strides=2,
        activation=activation,
    ))
    model.add(Conv2D(
        filters=64,
        kernel_size=3,
        activation=activation,
    ))
    model.add(Conv2D(
        filters=64,
        kernel_size=3,
        activation=activation
    ))
    model.add(Flatten())
    model.add(Dense(100, activation=activation))
    model.add(Dense(50, activation=activation))
    model.add(Dense(10, activation=activation))
    model.add(Dense(1))

    # Compile model
    optimizer = Adadelta()
    model.compile(
        loss='mean_squared_error',
        optimizer=optimizer
    )

    return model
