from net import net_utils

def Resnet(input, img_channel):
    temp = net_utils.resnet_start_block(input, img_channel)
    for i in range(3):
        temp = net_utils.resnet_identity_block(temp)
    temp = net_utils.resnet_end_block(temp, img_channel)
    output = net_utils.add(input, temp)

    return output
