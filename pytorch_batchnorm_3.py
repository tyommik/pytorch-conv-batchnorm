import torch
import torch.nn as nn


input_size = 3
batch_size = 5
eps = 1e-1


class CustomBatchNorm1d:
    def __init__(self, weight, bias, eps, momentum):
        self.weight = weight
        self.bias = bias
        self.eps = eps
        self.momentum = momentum

        self.running_var = None
        self.running_mean = None

        self.isEval = False

    def __call__(self, input_tensor):
        batch_size, input_len = input_tensor.shape

        if self.running_var is None or self.running_mean is None:
            self.running_var = torch.ones((1, input_tensor.shape[1]))
            self.running_mean = torch.zeros((1, input_tensor.shape[1]))

        if self.isEval:
            mean = self.running_mean
            var = self.running_var
        else:
            mean = input_tensor.mean(dim=0)
            var = input_tensor.var(dim=0, unbiased=False)

            self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
            self.running_var = self.momentum * var * batch_size / (batch_size-1) + (1 - self.momentum) * self.running_var

        normed_tensor = self.weight * (input_tensor - mean) / torch.sqrt(var + eps) + self.bias

        return normed_tensor

    def eval(self):
        self.isEval = True


batch_norm = nn.BatchNorm1d(input_size, eps=eps)
batch_norm.bias.data = torch.randn(input_size, dtype=torch.float)
batch_norm.weight.data = torch.randn(input_size, dtype=torch.float)
batch_norm.momentum = 0.5

custom_batch_norm1d = CustomBatchNorm1d(batch_norm.weight.data,
                                        batch_norm.bias.data, eps, batch_norm.momentum)

# Проверка происходит автоматически вызовом следующего кода
# (раскомментируйте для самостоятельной проверки,
#  в коде для сдачи задания должно быть закомментировано):
all_correct = True

for i in range(8):
    torch_input = torch.randn(batch_size, input_size, dtype=torch.float)
    norm_output = batch_norm(torch_input)
    custom_output = custom_batch_norm1d(torch_input)
    all_correct &= torch.allclose(norm_output, custom_output) \
        and norm_output.shape == custom_output.shape

print('=================================\n\n')
batch_norm.eval()
custom_batch_norm1d.eval()

for i in range(8):
    torch_input = torch.randn(batch_size, input_size, dtype=torch.float)
    norm_output = batch_norm(torch_input)
    custom_output = custom_batch_norm1d(torch_input)
    all_correct &= torch.allclose(norm_output, custom_output) \
        and norm_output.shape == custom_output.shape
print(all_correct)