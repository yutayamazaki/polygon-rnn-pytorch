import unittest
from typing import List, Tuple

import torch
import torch.nn as nn

from src import models


class PolygonVGG16Tests(unittest.TestCase):

    def test_forward_simple(self):
        net: nn.Module = models.PolygonVGG16(pretrained=False)
        inputs: torch.Tensor = torch.zeros((2, 3, 224, 224))
        out_list: List[torch.Tensor] = net(inputs)

        self.assertEqual(len(out_list), 4)
        self.assertIsInstance(out_list, list)

        expected_shape: List[torch.Size] = [
            torch.Size((2, 128, 56, 56)),
            torch.Size((2, 256, 28, 28)),
            torch.Size((2, 512, 28, 28)),
            torch.Size((2, 512, 14, 14))
        ]
        for out, shape in zip(out_list, expected_shape):
            self.assertEqual(out.size(), shape)

    def test_forward_assertion(self):
        net: nn.Module = models.PolygonVGG16(pretrained=False)
        invalid_inputs: torch.Tensor = torch.zeros((2, 3, 2, 3))

        with self.assertRaises(AssertionError):
            net(invalid_inputs)


class PolygonCNNTests(unittest.TestCase):

    def test_forward_simple(self):
        net: nn.Module = models.PolygonCNN(pretrained=False)
        inputs: torch.Tensor = torch.zeros((2, 3, 224, 224))
        outputs: torch.Tensor = net(inputs)

        expected_shape: torch.Size = torch.Size((2, 128, 28, 28))
        self.assertEqual(outputs.size(), expected_shape)

    def test_forward_assertion(self):
        net: nn.Module = models.PolygonCNN(pretrained=False)
        fuckin_inputs: torch.Tensor = torch.zeros((2, 3, 2, 1))

        with self.assertRaises(AssertionError):
            net(fuckin_inputs)


class ConvLSTMCellTests(unittest.TestCase):

    def test_forward_simple(self):
        input_size: Tuple[int, int] = (28, 28)  # height and width
        input_dim: int = 131  # 128 + 3
        hidden_dim: int = 32
        kernel_size: int = (3, 3)
        bias: bool = True
        batch: int = 2

        net = models.ConvLSTMCell(
            input_size,
            input_dim,
            hidden_dim,
            kernel_size,
            bias
        )
        inputs: torch.Tensor = torch.zeros((batch, input_dim, 28, 28))
        cur_state: List[torch.Tensor] = [
            torch.zeros((batch, hidden_dim, 28, 28)),
            torch.zeros((batch, hidden_dim, 28, 28))
        ]
        h, c = net(inputs, cur_state)

        self.assertEqual(h.size(), torch.Size((batch, hidden_dim, 28, 28)))
        self.assertEqual(c.size(), torch.Size((batch, hidden_dim, 28, 28)))


class ConvLSTMTests(unittest.TestCase):

    def test_forward_simple(self):
        input_size: Tuple[int, int] = (28, 28)  # height and width
        input_dim: int = 131  # 128 + 3
        hidden_dim: List[int, int] = [32, 8]
        kernel_size: Tuple[int] = (3, 3)
        bias: bool = True
        batch: int = 2
        num_layers: int = 2
        batch_first: bool = True
        return_all_layers: bool = True
        seq_len: int = 58

        net = models.ConvLSTM(
            input_size,
            input_dim,
            hidden_dim,
            kernel_size,
            num_layers,
            batch_first,
            bias,
            return_all_layers
        )
        # (B, seq_len, 128 + 3(first, sencond, third), H, W)
        inputs: torch.Tensor = torch.zeros((2, seq_len, 131, 28, 28))
        layer_output_list, last_state_list = net(inputs)

        self.assertIsInstance(layer_output_list, list)
        self.assertIsInstance(last_state_list, list)

        expected_shape = [
            torch.Size([2, 58, 32, 28, 28]),
            torch.Size([2, 58, 8, 28, 28])
        ]
        for out, shape in zip(layer_output_list, expected_shape):
            self.assertEqual(out.size(), shape)

        expected_shape = [
            torch.Size([2, 32, 28, 28]),
            torch.Size([2, 8, 28, 28])
        ]
        for out, shape in zip(last_state_list, expected_shape):
            for o in out:
                self.assertEqual(o.size(), shape)


class PolygonRNNTests(unittest.TestCase):

    def test_forward_simple(self):
        b, h, w, c, seq_len = 2, 224, 224, 3, 58
        img = torch.zeros((b, c, h, w))
        x1 = torch.zeros((b, 28*28+3))
        x2 = torch.zeros((b, seq_len, 28*28+3))
        x3 = torch.zeros((b, seq_len, 28*28+3))

        net = models.PolygonRNN(pretrained=False)
        out = net(img, x1, x2, x3)

        expected_shape: torch.Size = torch.Size((b, seq_len, 28*28+3))
        self.assertEqual(out.size(), expected_shape)

    def test_forward_assertion(self):
        b, h, w, c, seq_len = 2, 224, 2, 3, 58
        img: torch.Tensor = torch.zeros((b, c, h, w))
        x1: torch.Tensor = torch.zeros((b, 28*28+3))
        x2: torch.Tensor = torch.zeros((b, seq_len, 28*28+3))
        x3: torch.Tensor = torch.zeros((b, seq_len, 28*28+3))

        net = models.PolygonRNN(pretrained=False)
        with self.assertRaises(AssertionError):
            net(img, x1, x2, x3)
