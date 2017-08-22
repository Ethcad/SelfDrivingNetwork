from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import Adadelta
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D


def more_filters_model(blocks):
    # Rectified linear activation function
    activation = 'relu'

    # Padding to make output the same size as input
    padding = 'same'

    # Create a model with lots and lots of filters, inspired by the VGG series
    model = Sequential()

    # Input layer which does nothing
    model.add(ZeroPadding2D(input_shape=(200, 66, 3), padding=0))

    # Iterate over blocks matrix which contains the number of filters and number of convolutional layers for each block
    # as well as a flag telling us whether a pooling layer should be included or not
    for filters, conv_layers, include_pool in blocks:
        # Add multiple convolutional layers, configured in blocks
        for j in range(conv_layers):
            model.add(Conv2D(
                kernel_size=3,
                filters=filters,
                strides=2,
                padding=padding,
                activation=activation
            ))

        # Add a single max pooling layer
        if include_pool:
            model.add(MaxPooling2D(
                pool_size=2,
                strides=2
            ))

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(4096, activation=activation))
    model.add(Dense(4096, activation=activation))
    model.add(Dense(1))

    # Compile model
    optimizer = Adadelta()
    model.compile(
        loss='mean_squared_error',
        optimizer=optimizer
    )

    return model
