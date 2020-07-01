import argparse
import copy
import glob
import math
import os
import random
from collections import Counter
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
    utils.seed_everything(428)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-w', '--weights', type=str, help='The weights of trained model.'
    )
    args = parser.parse_args()

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    net: nn.Module = PolygonRNN()
    net.load_state_dict(torch.load(args.weights, map_location=device))
    net = net.eval()
    dtype = torch.FloatTensor
    dtype_t = torch.LongTensor

    images: List[str] = glob.glob('../leftImg8bit_trainvaltest/leftImg8bit/val/*/*.png')
    for img_path in images[:50]:
        fname: str = os.path.basename(img_path)
        file_id, _ = os.path.splitext(fname)
        file_id = file_id[:-12]
        city: str = file_id.split('_')[0]
        json_path: str = f'../gtFine_trainvaltest/gtFine/val/{city}/{file_id}_gtFine_polygons.json'
        dataset: List[Tuple[str, str]] = [(img_path, json_path)]

        demo_dataset = PolygonCityScapesDataset(
            city_paths=dataset, transform=transforms.ToTensor()
        )
        demo_loader = torch.utils.data.DataLoader(
            demo_dataset, batch_size=1
        )

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

        # Save prediction result
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        orig_image: Image.Image = transforms.ToPILImage()(x[0]).resize((224, 224))
        ax1.imshow(orig_image)
        ax1.set_title('Original')

        draw = ImageDraw.Draw(orig_image)
        s: float = 224 / 28
        polygon = [(p[0] * s, p[1] * s) for p in polygon]
        draw.line(xy=sort_polygon(polygon), fill=255)
        ax3.imshow(orig_image)
        ax3.set_title('Prediction')

        orig_image2: Image.Image = transforms.ToPILImage()(x[0]).resize((224, 224))
        draw = ImageDraw.Draw(orig_image2)
        polygon_anno: List[Tuple[int, int]] = [(s * index2point(int(idx))[0], s * index2point(int(idx))[1]) for idx in gt[0]]
        draw.line(xy=sort_polygon(polygon_anno), fill=255)
        ax2.imshow(orig_image2)
        ax2.set_title('Annotation')

        fig.savefig(f'./demo/{file_id}.png')
        plt.clf()

        del fig, ax1, ax2, ax3
