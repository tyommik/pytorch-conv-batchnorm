import torch
from abc import ABC, abstractmethod


def calc_out_shape(input_matrix_shape, out_channels, kernel_size, stride, padding):
    batch_size, channels_count, input_height, input_width = input_matrix_shape
    output_height = (input_height + 2 * padding - (kernel_size - 1) - 1) // stride + 1
    output_width = (input_width + 2 * padding - (kernel_size - 1) - 1) // stride + 1

    return batch_size, out_channels, output_height, output_width


class ABCConv2d(ABC):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def set_kernel(self, kernel):
        self.kernel = kernel

    @abstractmethod
    def __call__(self, input_tensor):
        pass


class Conv2d(ABCConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                      stride, padding=0, bias=False)

    def set_kernel(self, kernel):
        self.conv2d.weight.data = kernel

    def __call__(self, input_tensor):
        return self.conv2d(input_tensor)


def create_and_call_conv2d_layer(conv2d_layer_class, stride, kernel, input_matrix):
    out_channels = kernel.shape[0]
    in_channels = kernel.shape[1]
    kernel_size = kernel.shape[2]

    layer = conv2d_layer_class(in_channels, out_channels, kernel_size, stride)
    layer.set_kernel(kernel)

    return layer(input_matrix)


def test_conv2d_layer(conv2d_layer_class, batch_size=2,
                      input_height=4, input_width=4, stride=2):
    kernel = torch.tensor(
                      [[[[0., 1, 0],
                         [1,  2, 1],
                         [0,  1, 0]],

                        [[1, 2, 1],
                         [0, 3, 3],
                         [0, 1, 10]],

                        [[10, 11, 12],
                         [13, 14, 15],
                         [16, 17, 18]]]])


    in_channels = kernel.shape[1]

    input_tensor = torch.arange(0, batch_size * in_channels *
                                input_height * input_width,
                                out=torch.FloatTensor()) \
        .reshape(batch_size, in_channels, input_height, input_width)

    custom_conv2d_out = create_and_call_conv2d_layer(
        conv2d_layer_class, stride, kernel, input_tensor)
    conv2d_out = create_and_call_conv2d_layer(
        Conv2d, stride, kernel, input_tensor)

    return torch.allclose(custom_conv2d_out, conv2d_out)


# Сверточный слой через циклы.
class Conv2dLoop(ABCConv2d):
    def __call__(self, input_tensor):
        batch_size, out_channels, output_height, output_width = calc_out_shape(
            input_tensor.shape,
            self.out_channels,
            self.kernel_size,
            self.stride,
            padding=0)

        # создадим выходной тензор, заполненный нулями
        output_tensor = torch.zeros(batch_size, out_channels, output_height, output_width)

        # вычисление свертки с использованием циклов.
        # цикл по входным батчам(изображениям)
        for num_batch, batch in enumerate(input_tensor):

            # цикл по фильтрам (количество фильтров совпадает с количеством выходных каналов)
            for num_kernel, kernel in enumerate(self.kernel):

                # цикл по размерам выходного изображения
                for i in range(output_height):
                    for j in range(output_width):
                        # вырезаем кусочек из батча (сразу по всем входным каналам)
                        current_row = self.stride * i
                        current_column = self.stride * j
                        current_slice = batch[:, current_row:current_row + self.kernel_size,
                                        current_column:current_column + self.kernel_size]

                        # умножаем кусочек на фильтр
                        res = float((current_slice * kernel).sum())

                        # заполняем ячейку в выходном тензоре
                        output_tensor[num_batch, num_kernel, i, j] = res

        return output_tensor


class Conv2dMatrix(ABCConv2d):
    # Функция преобразование кернела в матрицу нужного вида.
    def _unsqueeze_kernel(self, torch_input, output_height, output_width):
        batch_size, out_channels, input_height, input_width = torch_input.shape
        matrix = []
        _, in_channels, _, _ = self.kernel.shape
        for i_kernel in range(in_channels):
            for height in range(output_height):
                for width in range(output_width):
                    # TODO сразу одной матрицей сделать
                    layer = torch.zeros((input_height, input_width))
                    layer[height * self.stride:height * self.stride + self.kernel_size,
                            width * self.stride:width * self.stride + self.kernel_size] = self.kernel[0, i_kernel]
                    matrix.append(layer.flatten().reshape(1, -1))

        kernel_unsqueezed = torch.cat(matrix, dim=0).reshape(-1, in_channels * input_height * input_width)

        return kernel_unsqueezed

    def __call__(self, torch_input):
        batch_size, out_channels, output_height, output_width\
            = calc_out_shape(
                input_matrix_shape=torch_input.shape,
                out_channels=self.kernel.shape[0],
                kernel_size=self.kernel.shape[2],
                stride=self.stride,
                padding=0)

        kernel_unsqueezed = self._unsqueeze_kernel(torch_input, output_height, output_width)
        image = torch_input.view((batch_size, -1)).permute(1, 0)
        result = kernel_unsqueezed @ image
        res = result.permute(1, 0).reshape((batch_size, self.out_channels,
                                          output_height, output_width))
        return res


class Conv2dMatrixV2(ABCConv2d):
    # Функция преобразования кернела в нужный формат.
    def _convert_kernel(self):
        kernel = self.kernel
        out_channels = kernel.shape[0]
        converted_kernel = self.kernel.reshape(out_channels, -1)
        return converted_kernel

    # Функция преобразования входа в нужный формат.
    def _convert_input(self, torch_input, output_height, output_width):
        batch_size, out_channels, input_height, input_width = torch_input.shape
        _, in_channels, kernel_height, kernel_width = self.kernel.shape

        out = torch.zeros((out_channels*kernel_height*kernel_width, batch_size))
        for batch in range(batch_size):
            arr = []
            for image_layer in range(in_channels):
                for height in range(output_height):
                    for width in range(output_width):
                        # TODO сразу одной матрицей сделать
                        res = torch_input[batch, image_layer, height * self.stride:height * self.stride + self.kernel_size,
                                width * self.stride:width * self.stride + self.kernel_size]
                        arr.append(res.flatten().reshape(-1, 1))
            res = torch.cat(arr, dim=0).squeeze()
            out[:, batch] = res
        return out

    def __call__(self, torch_input):
        batch_size, out_channels, output_height, output_width\
            = calc_out_shape(
                input_matrix_shape=torch_input.shape,
                out_channels=self.kernel.shape[0],
                kernel_size=self.kernel.shape[2],
                stride=self.stride,
                padding=0)

        converted_kernel = self._convert_kernel()
        converted_input = self._convert_input(torch_input, output_height, output_width)

        conv2d_out_alternative_matrix_v2 = converted_kernel @ converted_input
        return conv2d_out_alternative_matrix_v2.transpose(0, 1).view(torch_input.shape[0],
                                                     self.out_channels, output_height,
                                                     output_width).transpose(1, 3).transpose(2, 3)


# Проверка происходит автоматически вызовом следующего кода
# (раскомментируйте для самостоятельной проверки,
#  в коде для сдачи задания должно быть закомментировано):
print(test_conv2d_layer(Conv2dLoop))
print(test_conv2d_layer(Conv2dMatrix))
print(test_conv2d_layer(Conv2dMatrixV2))