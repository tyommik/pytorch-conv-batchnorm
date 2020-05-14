import torch
import torch.nn as nn
import numpy as np

channel_count = 6
eps = 1e-3
batch_size = 20
input_size = 2

input_tensor = torch.randn(batch_size, channel_count, input_size)


def custom_group_norm(input_tensor, groups, eps):
    batch, channels, length = input_tensor.shape
    normed_tens = torch.zeros(input_tensor.shape)
    groups  = int(channel_count / groups)
    for i in range(batch):
        for j in range(0, channel_count, groups):
            mean = input_tensor[i, slice(0 + j, j + groups)].mean()
            var = input_tensor[i, slice(0 + j, j + groups)].var(unbiased=False)

            normed_tens[i, slice(0 + j, j + groups)] = (input_tensor[i, slice(0 + j, j + groups)] - mean) / torch.sqrt(var + eps)

    return normed_tens


# Проверка происходит автоматически вызовом следующего кода
# (раскомментируйте для самостоятельной проверки,
#  в коде для сдачи задания должно быть закомментировано):
all_correct = True
for groups in [1, 2, 3, 6]:
    group_norm = nn.GroupNorm(groups, channel_count, eps=eps, affine=False)
    norm_output = group_norm(input_tensor)
    custom_output = custom_group_norm(input_tensor, groups, eps)

    # print('Input: \n')
    # print(input_tensor)
    print('\nNorm:')
    print(norm_output)
    print('\nCustom: ')
    print(custom_output)

    all_correct &= torch.allclose(norm_output, custom_output, 1e-3)
    all_correct &= norm_output.shape == custom_output.shape

print(all_correct)