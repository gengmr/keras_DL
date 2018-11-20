import keras
from net import net_utils

def Unet(input, img_channel):
    x_0 = net_utils.conv(input)
    x_1 = net_utils.conv(x_0)
    x_2 = net_utils.conv(x_1)
    x_3 = net_utils.conv(x_2)
    x_4 = net_utils.conv(x_3)
    x0 = net_utils.conv(x_4)
    x0 = keras.layers.concatenate([x0, x_4], -1)
    x1 = net_utils.conv(x0)
    x1 = keras.layers.concatenate([x1, x_3], -1)
    x2 = net_utils.conv(x1)
    x2 = keras.layers.concatenate([x2, x_2], -1)
    x3 = net_utils.conv(x2)
    x3 = keras.layers.concatenate([x3, x_1], -1)
    x4 = net_utils.conv(x3)
    x4 = keras.layers.concatenate([x4, x_0], -1)
    output = net_utils.conv(x4, out_channel=img_channel)
    output = keras.layers.Add()([output, input])

    return output
