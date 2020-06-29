import copy
import math
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torch.autograd import Variable
from torchvision import transforms

from datasets import PolygonCityScapesDataset
from models import PolygonRNN
from train import load_yaml
import utils

batch_size: int = 6
lr: float = 1e-4
num_epochs: int = 2
seq_len: int = 60


def index2point(index: int, height: int = 28, width: int = 28) -> Tuple[int, int]:
    num_rows: int = index // width
    num_cols: int = index % width
    if num_rows >= 28:
        num_rows = 27
    if num_cols >= 28:
        num_cols = 27
    return (num_rows, num_cols)


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def sort_polygon(polygon: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    copied_polygon: List[Tuple[int, int]] = list(set(polygon))
    sorted_polygon: List[Tuple[int, int]] = []

    x, y = polygon[0]
    while copied_polygon:
        nearest_index: int = -1
        nearest_distance: float = 1e9
        for idx, (x_i, y_i) in enumerate(copied_polygon):
            if (x == x_i) and (y == y_i):
                continue
            if (x_i, y_i) in sorted_polygon:
                continue
            distance = euclidean_distance(x, y, x_i, y_i)
            if nearest_distance > distance:
                nearest_distance = distance
                nearest_index = copy.deepcopy(idx)

        sorted_polygon.append(copied_polygon[nearest_index])
        x, y = copied_polygon[nearest_index]
        copied_polygon.pop(nearest_index)
    return sorted_polygon


if __name__ == '__main__':
    from collections import Counter

    utils.seed_everything()

    img_path: str = '../leftImg8bit_trainvaltest/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png'
    json_path: str = '../gtFine_trainvaltest/gtFine/train/aachen/aachen_000000_000019_gtFine_polygons.json'
    dataset: List[Tuple[str, str]] = [(img_path, json_path)]

    demo_dataset = PolygonCityScapesDataset(
        city_paths=dataset, transform=transforms.ToTensor()
    )
    demo_loader = torch.utils.data.DataLoader(
        demo_dataset, batch_size=1
    )

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    net: nn.Module = PolygonRNN()
    net.load_state_dict(torch.load('../experiments/20200626_04-41-59/weights/loss3.62720_epoch045.pth', map_location=device))
    net = net.eval()

    dtype = torch.FloatTensor
    dtype_t = torch.LongTensor

    for data in demo_loader:
        x = Variable(data[0].type(dtype))
        x1 = Variable(data[1].type(dtype))
        x2 = Variable(data[2].type(dtype))
        x3 = Variable(data[3].type(dtype))
        gt = Variable(data[4].type(dtype_t))

        outputs: torch.Tensor = net(x, x1, x2, x3)

    output: np.ndarray = outputs[0].detach().numpy()  # (seq_len, 787)
    polygon: List[Tuple[int, int]] = []

    for out in output:
        index: int = int(np.argmax(out))
        point = index2point(index)
        polygon.append(point)

    polygon = list(set(polygon))

    # img = Image.fromarray(np.uint8(np.zeros((28, 28))))
    img = np.zeros((28, 28))
    for x_, y_ in polygon:
        img[x_, y_] = 1.
    # draw = ImageDraw.Draw(img)
    # draw.polygon(xy=sort_polygon(polygon), fill=255)

    # Save prediction result
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.imshow(np.array(img))
    ax1.set_title('Prediction')
    img = np.uint8(x[0].permute(2, 1, 0).detach().numpy() * 255)

    ax2.imshow(Image.fromarray(img).resize((28, 28)))
    ax2.set_title('Original')

    fig.savefig('demo.png')
