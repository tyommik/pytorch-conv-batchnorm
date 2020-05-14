import torch
import torch.nn as nn

eps = 1e-3

input_channels = 3
batch_size = 3
height = 10
width = 10

batch_norm_2d = nn.BatchNorm2d(input_channels, affine=False, eps=eps)

input_tensor = torch.randn(batch_size, input_channels, height, width, dtype=torch.float)

# arange
input_tensor = torch.arange(3 * 3 * 10 * 10, dtype=torch.float).reshape(input_tensor.shape)
#
# one_channel = torch.zeros((3, 3)) + 1
#
# two_channel = torch.zeros((3, 3)) + 2
#
# three_channel = torch.zeros((3, 3)) + 3
#
# one_img = torch.stack([one_channel, one_channel, one_channel])
# two_img = torch.stack([two_channel, two_channel, two_channel])
# three_img = torch.stack([three_channel, three_channel, three_channel])
# input_tensor = torch.stack([one_img, two_img, three_img])



def custom_batch_norm2d(input_tensor, eps):
    images, channels, width, height = input_tensor.shape
    normed_tensor = torch.zeros(input_tensor.shape)

    for channel_idx in range(channels):
        mean = input_tensor[:, channel_idx].mean()
        var = input_tensor[:, channel_idx].var(unbiased=False)

        normed_tensor[:, channel_idx] = (input_tensor[:, channel_idx] - mean) / torch.sqrt(var + eps)
    return normed_tensor


# Проверка происходит автоматически вызовом следующего кода
# (раскомментируйте для самостоятельной проверки,
#  в коде для сдачи задания должно быть закомментировано):
norm_output = batch_norm_2d(input_tensor)
print('Torch\n')
print(norm_output)

print('\n\nCustom\n')
custom_output = custom_batch_norm2d(input_tensor, eps)
print(custom_output)
print(torch.allclose(norm_output, custom_output) and norm_output.shape == custom_output.shape)