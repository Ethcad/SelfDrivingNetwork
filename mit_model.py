from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import Adadelta
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D


def mit_model():
    # Hyperbolic tangent activation function
    activation = 'relu'

    # Create the model
    model = Sequential()
    model.add(ZeroPadding2D(input_shape=(200, 66, 3), padding=2))
    model.add(Conv2D(
        input_shape=(200, 66, 3),
        kernel_size=3,
        filters=8,
        strides=3,
        activation=activation
    ))
    model.add(ZeroPadding2D(padding=2))
    model.add(Conv2D(
        kernel_size=3,
        filters=8,
        strides=3,
        activation=activation
    ))
    model.add(ZeroPadding2D(padding=2))
    model.add(Conv2D(
        kernel_size=3,
        filters=8,
        strides=3,
        activation=activation
    ))
    model.add(MaxPooling2D(
        pool_size=4,
        strides=2
    ))
    model.add(Flatten())
    model.add(Dense(units=1))

    # Compile model
    optimizer = Adadelta()
    model.compile(
        loss='mean_squared_error',
        optimizer=optimizer
    )

    return model
