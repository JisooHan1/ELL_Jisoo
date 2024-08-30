def conv_par(kernel, in_channel, out_channel):
    bn_conv = 2*in_channel + ((kernel*kernel * in_channel + 1) * out_channel)
    return bn_conv

def densenet_bundle_par(initial_channel):
    pars = 0
    for i in range(16):
        pars += conv_par(1, initial_channel+(12*i), 48) + conv_par(3, 48, 12)
    return pars

# resnet
conv1 = conv_par(3, 3, 64)
bundle1 = 4 * conv_par(3, 64, 64)
bundle2 = (conv_par(3, 64, 128) + conv_par(3, 128, 128) + conv_par(1, 64, 128)) + (2 * conv_par(3, 128, 128))
bundle3 = (conv_par(3, 128, 256) + conv_par(3, 256, 256) + conv_par(1, 128, 256)) + (2 * conv_par(3, 256, 256))
bundle4 = (conv_par(3, 256, 512) + conv_par(3, 512, 512) + conv_par(1, 256, 512)) + (2 * conv_par(3, 512, 512))
fc = (512 + 1)*10

resnet_par = conv1 + bundle1 + bundle2 + bundle3 + bundle4 + fc
print("number of parameter of ResNet18: ", resnet_par)

# DensNet
Conv1 = conv_par(3, 3, 24)
Bundle1 = densenet_bundle_par(24)
Trans1 = conv_par(1, 216, 108)
Bundle2 = densenet_bundle_par(108)
Trans2 = conv_par(1, 300, 150)
Bundle3 = densenet_bundle_par(150)
Fc = (342 + 1)*10

densenet_par = Conv1 + Bundle1 + Bundle2 + Bundle3 + Trans1 + Trans2 + Fc
print("number of parameter of DenseNet_100_12: ", densenet_par)

