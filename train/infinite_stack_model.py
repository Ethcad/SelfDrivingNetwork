from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import Adadelta
from keras.layers.convolutional import Conv2D


def infinite_stack_model(num_conv_layers, num_dense_layers):
    # Rectified linear activation function
    activation = 'relu'

    # Padding to make output the same size as input
    padding = 'same'

    # Create a model with as many layers as possible, using a stride of only 1
    model = Sequential()

    # Convolutional layers
    for i in range(num_conv_layers):
        model.add(Conv2D(
            input_shape=(200, 66, 3),
            filters=4,
            strides=1,
            kernel_size=2,
            padding=padding,
            activation=activation
        ))

    # Fully connected layers
    model.add(Flatten())
    for i in range(num_dense_layers):
        model.add(Dense(1024, activation=activation))
    model.add(Dense(1))

    # Compile model
    optimizer = Adadelta()
    model.compile(
        loss='mean_squared_error',
        optimizer=optimizer
    )

    return model
