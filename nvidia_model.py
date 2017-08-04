from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import Adadelta
from keras.layers.convolutional import Conv2D


def nvidia_model():
    # Hyperbolic tangent activation function
    activation = 'tanh'

    # Create the model
    model = Sequential()
    model.add(Conv2D(
        input_shape=(200, 66, 3),
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
        activation=activation,
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

    print(model.summary())
    return model
