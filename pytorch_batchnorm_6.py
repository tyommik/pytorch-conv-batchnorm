import torch
import torch.nn as nn

eps = 1e-3

batch_size = 5
input_channels = 2
input_length = 30

instance_norm = nn.InstanceNorm1d(input_channels, affine=False, eps=eps)

input_tensor = torch.randn(batch_size, input_channels, input_length, dtype=torch.float)


def custom_instance_norm1d(input_tensor, eps):
    dim_count = len(input_tensor.shape) - 1
    input_tensor_shape = input_tensor.shape

    mean = input_tensor.mean(dim=dim_count, keepdim=True)
    var = input_tensor.var(dim=dim_count, unbiased=False, keepdim=True)

    normed_tensor = (input_tensor - mean) / torch.sqrt(var + eps) * 1 + 0
    return normed_tensor.reshape(*input_tensor_shape)


# Проверка происходит автоматически вызовом следующего кода
# (раскомментируйте для самостоятельной проверки,
#  в коде для сдачи задания должно быть закомментировано):
norm_output = instance_norm(input_tensor)
custom_output = custom_instance_norm1d(input_tensor, eps)

print('Input: \n')
print(input_tensor)
print('\nNorm:')
print(norm_output)
print('\nCustom: ')
print(custom_output)

print(torch.allclose(norm_output, custom_output) and norm_output.shape == custom_output.shape)