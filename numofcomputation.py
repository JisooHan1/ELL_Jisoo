def conv_comp(kernel, input_size, in_channel, out_channel):
    conv = (kernel*kernel * in_channel)*(input_size*input_size) * out_channel
    return conv

def densenet_bundle_comp(input_size, initial_channel):
    pars = 0
    for i in range(16):
        pars += conv_comp(1, input_size, initial_channel+(12*i), 48) + conv_comp(3, input_size, 48, 12)
    return pars

# resnet
conv1 = conv_comp(kernel=3, input_size=32, in_channel=3, out_channel=64)
bundle1 = 4 * conv_comp(kernel=3, input_size=32, in_channel=64, out_channel=64)
bundle2 = (conv_comp(kernel=3, input_size=32, in_channel=64, out_channel=128)
           + 3 * conv_comp(kernel=3, input_size=16, in_channel=128, out_channel=128)) + (conv_comp(kernel=1, input_size=32, in_channel=64, out_channel=128))
bundle3 = (conv_comp(kernel=3, input_size=16, in_channel=128, out_channel=256)
           + 3 * conv_comp(kernel=3, input_size=8, in_channel=256, out_channel=256)) + (conv_comp(kernel=1, input_size=16, in_channel=128, out_channel=256))
bundle4 = (conv_comp(kernel=3, input_size=8, in_channel=256, out_channel=512)
           + 3 * conv_comp(kernel=3, input_size=4, in_channel=512, out_channel=512)) + (conv_comp(kernel=1, input_size=8, in_channel=256, out_channel=512))
fc = 512*10 * 2

resnet_par = conv1 + bundle1 + bundle2 + bundle3 + bundle4 + fc
print("number of computation of ResNet18: ", resnet_par)

# DensNet
Conv1 = conv_comp(kernel=3, input_size=32, in_channel=3, out_channel=24)
Bundle1 = densenet_bundle_comp(input_size=32, initial_channel=24)
Trans1 = conv_comp(kernel=3, input_size=32, in_channel=216, out_channel=108)
Bundle2 = densenet_bundle_comp(input_size=16, initial_channel=108)
Trans2 = conv_comp(kernel=3, input_size=16, in_channel=300, out_channel=150)
Bundle3 = densenet_bundle_comp(input_size=8, initial_channel=150)
Fc = 342*10 * 2

densenet_par = Conv1 + Bundle1 + Bundle2 + Bundle3 + Trans1 + Trans2 + Fc
print("number of computation of DenseNet_100_12: ", densenet_par)

