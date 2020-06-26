import unittest
from typing import List

import torch
import torch.nn as nn

from src import models


class PolygonVGG16Tests(unittest.TestCase):

    def setUp(self):
        self.docs: List[str] = ['I am a pen', 'I']

    def test_forward_simple(self):
        net: nn.Module = models.PolygonVGG16()
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
