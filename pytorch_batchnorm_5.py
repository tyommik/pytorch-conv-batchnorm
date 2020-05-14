import torch
import torch.nn as nn


eps = 1e-10
# def custom_layer_norm_v(input_tensor, eps):
#     normed_tensor = torch.zeros(input_tensor.shape)
#
#     mean = input_tensor.mean(dim=1)
#     print('mean: ', mean.shape)
#
#     var = input_tensor.var(dim=1, unbiased=False)
#     print('var: ', var.shape)
#     for channel in range(4):
#         normed_tensor[:, channel] = (input_tensor[:, channel] - mean) / torch.sqrt(var + eps)
#
#     # for channel_idx in range(channels):
#     #     mean = input_tensor[:, channel_idx].mean()
#     #     var = input_tensor[:, channel_idx].var(unbiased=False)
#     #
#     #     normed_tensor[:, c

def custom_layer_norm(input_tensor, eps):
    normed_tensor = torch.zeros(input_tensor.shape)
    len_dim = len(input_tensor.shape)
    mean = input_tensor.mean(dim=list(range(1, len_dim)))
    print('mean: ', mean.shape)

    var = input_tensor.var(dim=list(range(1, len_dim)), unbiased=False)
    var = input_tensor.view(1, 3, -1).var(-1, unbiased=False)[0]
    print('var: ', var.shape)


    # mean = input_tensor.mean(dim=(1,2))
    # var = torch.tensor(input_tensor.numpy().var(axis=(1,2)))
    # for j in range(3):
    #     for i in range(4):
    #         normed_tensor[j,i,:] = (input_tensor[j, i,:] - mean[j])/(var[j] + eps).sqrt()
    for j in range(3):
        normed_tensor[j,:] = (input_tensor[j, :] - mean[j])/(var[j] + eps).sqrt()
    return normed_tensor


# Проверка происходит автоматически вызовом следующего кода
# (раскомментируйте для самостоятельной проверки,
#  в коде для сдачи задания должно быть закомментировано):
all_correct = True
for dim_count in range(3, 9):
    input_tensor = torch.randn(*list(range(3, dim_count + 2)), dtype=torch.float)
    print(input_tensor.shape)
    layer_norm = nn.LayerNorm(input_tensor.size()[1:], elementwise_affine=False, eps=eps)

    print('Input: \n')
    print(input_tensor)
    norm_output = layer_norm(input_tensor)
    print('\nNorm:')
    print(norm_output)
    custom_output = custom_layer_norm(input_tensor, eps)
    print('\nCustom: ')
    print(custom_output)

    all_correct &= torch.allclose(norm_output, custom_output, 1e-2)
    all_correct &= norm_output.shape == custom_output.shape

print(all_correct)