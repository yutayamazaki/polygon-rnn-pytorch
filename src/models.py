from typing import List

import torch
import torch.nn as nn
import torchvision.models
from torch.autograd import Variable


class PolygonVGG16(nn.Module):

    def __init__(self):
        super(PolygonVGG16, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=True).features
        # remove max-pooling
        self.layers = nn.Sequential(*list(vgg.children())[:-1])
        self.store_indices: List[int] = [9, 16, 22, 29]

        # used to assert
        self.output_shapes = (
            torch.Size([128, 56, 56]),
            torch.Size([256, 28, 28]),
            torch.Size([512, 28, 28]),
            torch.Size([512, 14, 14])
        )

    def _check_input(self, x: torch.Tensor):
        assert x.size()[1:] == torch.Size([3, 224, 224])

    def _check_output(self, x: List[torch.Tensor]):
        for out, expected in zip(x, self.output_shapes):
            assert out.size()[1:] == expected

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        self._check_input(x)

        outputs: List[torch.Tensor] = []
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx in self.store_indices:
                outputs.append(x)

        self._check_output(outputs)
        return outputs


class PolygonCNN(nn.Module):
    """This class concatenate outputs of VGG16Net and prepare inputs for
       ConvLSTM.
    """

    def __init__(self):
        super(PolygonCNN, self).__init__()
        self.cnn = PolygonVGG16()

        # x: (128, 56, 56) -> pool: (128, 28, 28) -> conv: (128, 28, 28)
        self.out1_layers = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3,
                stride=1, padding=1
            )
        )
        # x: (256, 28, 28) -> conv: (128, 28, 28)
        self.out2_layers = nn.Conv2d(
            in_channels=256, out_channels=128, kernel_size=3,
            stride=1, padding=1
        )
        # x: (512, 28, 28) -> conv: (128, 28, 28)
        self.out3_layers = nn.Conv2d(
            in_channels=512, out_channels=128, kernel_size=3,
            stride=1, padding=1
        )
        # x: (512, 14, 14) -> conv: (128, 14, 14) -> upsample: (128, 28, 28)
        self.out4_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=128, kernel_size=3,
                stride=1, padding=1
            ),
            nn.Upsample(size=28, mode='bilinear', align_corners=False)
        )

        # x: (512, 28, 28) -> conv: (128, 28, 28)
        self.conv = nn.Conv2d(
            in_channels=512, out_channels=128, kernel_size=3,
            stride=1, padding=1
        )

    def _check_input(self, x: torch.Tensor):
        assert x.size()[1:] == torch.Size([3, 224, 224])

    def _check_output(self, output: torch.Tensor):
        assert output.size()[1:] == torch.Size([128, 28, 28])

    def forward(self, x):
        self._check_input(x)

        x1, x2, x3, x4 = self.cnn(x)
        out1 = self.out1_layers(x1)  # (128, 28, 28)
        out2 = self.out2_layers(x2)  # (128, 28, 28)
        out3 = self.out3_layers(x3)  # (128, 28, 28)
        out4 = self.out4_layers(x4)  # (128, 28, 28)

        x_concat = torch.cat((out1, out2, out3, out4), 1)  # (512, 28, 28)
        out = self.conv(x_concat)  # (128, 28, 28)
        self._check_output(out)
        return out


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_conv, self.hidden_dim, dim=1
        )
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        return (
            Variable(
                torch.zeros(
                    batch_size, self.hidden_dim, self.height, self.width
                )
            ),
            Variable(
                torch.zeros(
                    batch_size, self.hidden_dim, self.height, self.width
                )
            )
        )


class ConvLSTM(nn.Module):

    def __init__(
        self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
        batch_first=False, bias=True, return_all_layers=False
    ):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are
        # lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = \
                self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(
                input_size=(self.height, self.width),
                input_dim=cur_input_dim,
                hidden_dim=self.hidden_dim[i],
                kernel_size=self.kernel_size[i],
                bias=self.bias
            ))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            pass
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=[h, c]
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        is_tuple: bool = isinstance(kernel_size, tuple)
        is_list: bool = isinstance(kernel_size, list)
        is_elem_tuple = all([isinstance(elem, tuple) for elem in kernel_size])
        is_ok: bool = is_tuple or is_list and is_elem_tuple
        if not is_ok:
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class PolygonRNN(nn.Module):

    def __init__(self):
        super(PolygonRNN, self).__init__()
        self.cnn = PolygonCNN()
        self.conv_lstm = ConvLSTM(
            input_size=(28, 28),
            input_dim=131,
            hidden_dim=[32, 8],
            kernel_size=(3, 3),
            num_layers=2,
            batch_first=True,
            bias=True,
            return_all_layers=True
        )
        self.lstm = nn.LSTM(
            input_size=28 * 28 * 8 + (28 * 28 + 3) * 2,
            hidden_size=28 * 28 * 2,
            batch_first=True
        )
        self.fc = nn.Linear(28 * 28 * 2, 28 * 28 + 3)

    def _check_input(self, x: torch.Tensor):
        assert x.size()[1:] == torch.Size([3, 224, 224])

    def forward(self, img, first, second, third):
        size: torch.Size = second.size()
        batch_size: int = size[0]
        seq_len: int = size[1]  # sequential length

        self._check_input(img)
        features = self.cnn(img)

        # (B, C: 128, H: 28, W: 28) -> (B, 1, C: 128, H: 28, W: 28)
        output = features.unsqueeze(1)
        # (B, 1, C* 128, H: 28, W: 28) -> (B, seq_len, C: 128, H: 28, W: 28)
        output = output.repeat(1, seq_len, 1, 1, 1)
        # (B, H*W*C: 28*28+3) -> (B, seq_len-1, 1, H: 28, W: 28)
        input_f = first[:, :-3].view(-1, 1, 28, 28).unsqueeze(1).repeat(
            1, seq_len - 1, 1, 1, 1
        )
        padding_f = torch.zeros([batch_size, 1, 1, 28, 28])

        # (B, seq_len-1, 1, 28, 28) -> (B, seq_len, 1, 28, 28)
        input_f = torch.cat([padding_f, input_f], dim=1)
        # (B, seq_len, 28*28+3) -> (B, seq_len, 1, 28, 28)
        input_s = second[:, :, :-3].view(-1, seq_len, 1, 28, 28)
        # (B, seq_len, 28*28+3) -> (B, seq_len, 1, 28, 28)
        input_t = third[:, :, :-3].view(-1, seq_len, 1, 28, 28)
        # (B, seq_len, 131, 28, 28)
        output = torch.cat([output, input_f, input_s, input_t], dim=2)
        output = self.conv_lstm(output)[0][-1]  # (B, seq_len, 8, 28, 28)

        # (B, seq_len, 8, 28, 28) -> (B, seq_len, 6272)
        output = output.contiguous().view(batch_size, seq_len, -1)
        # (B, seq_len, 7846)
        output = torch.cat([output, second, third], dim=2)
        output = self.lstm(output)[0]  # (B, seq_len, 1568)
        # (B*seq_len, 1568)
        output = output.contiguous().view(batch_size * seq_len, -1)
        output = self.fc(output)  # (B*seq_len, 787)
        output = output.contiguous().view(batch_size, seq_len, -1)

        return output  # (B, seq_len, 787)


if __name__ == '__main__':
    img = torch.zeros((2, 3, 224, 224))  # (B, C, H, W)
    first = torch.zeros((2, 28*28+3))
    second = torch.zeros((2, 58, 28*28+3))
    third = torch.zeros((2, 58, 28*28+3))

    model = PolygonRNN()
    out = model(img, first, second, third)
    print(out.size())
