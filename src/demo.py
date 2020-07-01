import argparse
import glob
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torch.autograd import Variable
from torchvision import transforms

import utils
from datasets import PolygonCityScapesDataset
from models import PolygonRNN
from polygon_tools import index2point, sort_polygon

batch_size: int = 6
lr: float = 1e-4
num_epochs: int = 2
seq_len: int = 60

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

    images: List[str] = glob.glob(
        '../leftImg8bit_trainvaltest/leftImg8bit/val/*/*.png'
    )
    for img_path in images[:10]:
        fname: str = os.path.basename(img_path)
        file_id, _ = os.path.splitext(fname)
        file_id = file_id[:-12]
        city: str = file_id.split('_')[0]
        json_path: str = f'../gtFine_trainvaltest/gtFine/val/{city}/' + \
                         f'{file_id}_gtFine_polygons.json'
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

        out_shape: Tuple[int, int] = (224, 224)
        orig_image: Image.Image = transforms.ToPILImage()(x[0]).resize(
            out_shape
        )
        ax1.imshow(orig_image)
        ax1.set_title('Original')

        draw = ImageDraw.Draw(orig_image)
        s: float = 224 / 28
        polygon_scaled: List[Tuple[int, int]] = [
            (int(p[0] * s), int(p[1] * s)) for p in polygon
        ]
        draw.line(xy=sort_polygon(polygon_scaled), fill=255)
        ax3.imshow(orig_image)
        ax3.set_title('Prediction')

        orig_image2: Image.Image = transforms.ToPILImage()(x[0]).resize(
            out_shape
        )
        draw = ImageDraw.Draw(orig_image2)
        polygon_anno: List[Tuple[int, int]] = [
            (
                int(s * index2point(int(idx))[0]),
                int(s * index2point(int(idx))[1])
            ) for idx in gt[0]
        ]
        draw.line(xy=sort_polygon(polygon_anno), fill=255)
        ax2.imshow(orig_image2)
        ax2.set_title('Annotation')

        fig.savefig(f'./demo/{file_id}.png')
        plt.clf()
