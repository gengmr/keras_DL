import keras
from keras.layers import Activation, Conv2D

def resnet_start_block(input, channel):
    out = Conv2D(filters=64,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding='same',
                 input_shape=(None, None, channel),  # 不限制输入图像维度
                 )(input)
    out = Activation('relu')(out)

    return out


def resnet_identity_block(input):
    out = Conv2D(filters=64,kernel_size=(1, 1),strides=(1, 1),padding='same')(input)
    out = Activation('relu')(out)

    out = Conv2D(filters=64,kernel_size=(3, 3),strides=(1, 1),padding='same')(out)
    out = Activation('relu')(out)

    out = Conv2D(filters=64,kernel_size=(1, 1),strides=(1, 1),padding='same')(out)

    # merge
    out = keras.layers.Add()([out, input])
    out = Activation('relu')(out)

    return out

def resnet_end_block(input, channel):
    out = Conv2D(filters=64,kernel_size=(1, 1),strides=(1, 1),padding='same')(input)
    out = Activation('relu')(out)

    out = Conv2D(filters=64,kernel_size=(3, 3),strides=(1, 1),padding='same')(out)
    out = Activation('relu')(out)

    out = Conv2D(filters=channel,kernel_size=(1, 1),strides=(1, 1),padding='same')(out)

    conv_input = Conv2D(filters=channel,kernel_size=(1, 1),strides=(1, 1),padding='same')(input)

    # merge
    out = keras.layers.Add()([out, conv_input])

    return out

def add(input1, input2):
    return keras.layers.Add()([input1, input2])


def conv(input, out_channel=64):
    out = Conv2D(filters=out_channel,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding='same'
                 )(input)

    return out