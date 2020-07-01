import glob
import json
import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

import polygon_tools as pt


def load_cityscapes(
    img_root: str = '../leftImg8bit_trainvaltest/',
    anno_root: str = '../gtFine_trainvaltest/'
) -> Dict[str, List[Tuple[str, str]]]:
    # train: 2975, val: 500, test: 1525
    datasets: Dict[str, List[Tuple[str, str]]] = {
        'train': [], 'val': [], 'test': []
    }
    for phase in ('train', 'val', 'test'):
        json_paths: List[str] = glob.glob(
            os.path.join(anno_root, 'gtFine', phase, '*', '*.json')
        )
        for json_path in json_paths:
            # file_id = 'munich_000187_000019_gtFine_'
            file_id: str = os.path.basename(json_path)[:-21]
            cityname: str = file_id.split('_')[0]

            img_path = os.path.join(
                img_root, 'leftImg8bit', phase, cityname,
                f'{file_id}_leftImg8bit.png'
            )

            if os.path.exists(img_path) and os.path.exists(json_path):
                datasets[phase].append((img_path, json_path))

        assert len(json_paths) == len(datasets[phase])
    return datasets
    # return {k: v[:6] for k, v in datasets.items()}


class PolygonCityScapesDataset(Dataset):

    def __init__(
        self,
        city_paths: List[Tuple[str, str]],
        seq_len: int = 60,
        transform=None
    ):
        self.city_paths: List[Tuple[str, str]] = city_paths
        self.seq_len: int = seq_len
        self.transform = transform
        self.class_names: List[str] = [
            'bike', 'bus', 'car', 'person', 'mbike', 'truck', 'rider', 'train'
        ]

    def __getitem__(self, index):
        img_path, json_path = self.city_paths[index]

        with open(json_path, 'r') as f:
            json_file: Dict[str, Any] = json.load(f)

        while True:
            # for obj in json_file['objects']:
            obj = random.choice(json_file['objects'])
            class_name: str = obj['label']
            polygon: List[List[int]] = obj['polygon']
            if class_name in self.class_names:
                break

        img: Image.Image = Image.open(img_path).convert('RGB')
        img, _ = pt.crop_object(img, polygon, (224, 224))

        point_num: int = len(polygon)
        point_count: int = 2
        label_array: np.ndarray = np.zeros([self.seq_len, 28 * 28 + 3])
        label_index_array: np.ndarray = np.zeros([self.seq_len])
        if point_num < self.seq_len - 3:
            polygon: List[List[float]] = pt.resize_polygon(polygon, (224, 224))
            polygon = np.array(polygon)
            for points in polygon:
                index_w: int = int(points[0] / 8)
                index_h: int = int(points[1] / 8)
                index: int = index_h + index_w * 28
                label_array[point_count, index] = 1
                label_index_array[point_count] = index
                point_count += 1

            label_array[point_count, 28 * 28] = 1
            label_index_array[point_count] = 28 * 28
            for kkk in range(point_count + 1, self.seq_len):
                if kkk % (point_num + 3) == point_num + 2:
                    index: int = 28 * 28
                elif kkk % (point_num + 3) == 0:
                    index: int = 28 * 28 + 1
                elif kkk % (point_num + 3) == 1:
                    index: int = 28 * 28 + 2
                else:
                    index_w = int(polygon[kkk % (point_num + 3) - 2][0] / 8)
                    index_h = int(polygon[kkk % (point_num + 3) - 2][1] / 8)
                    index: int = index_h + index_w * 28
                label_array[kkk, index] = 1
                label_index_array[kkk] = index

        else:
            scale: float = point_num * 1.0 / (self.seq_len - 3)
            index_list = (np.arange(0, self.seq_len - 3) * scale).astype(int)
            polygon: List[List[float]] = pt.resize_polygon(polygon, (224, 224))
            polygon = np.array(polygon)
            for points in polygon[index_list]:
                index_w: int = int(points[0] / 8)
                index_h: int = int(points[1] / 8)
                index: int = index_h + index_w * 28
                label_array[point_count, index] = 1
                label_index_array[point_count] = index
                point_count += 1
            for kkk in range(point_count, self.seq_len):
                index: int = 28 * 28
                label_array[kkk, index] = 1
                label_index_array[kkk] = index

        img: torch.Tensor = self.transform(img)
        # label_array[2] (787, ): {0: 786, 1: 1}
        # label_index_array (seq_len, ): Hold points of polygon in each
        #                                sequence by its index.
        return (
            img, label_array[2], label_array[:-2],
            label_array[1:-1], label_index_array[2:]
        )

    def __len__(self) -> int:
        # Number of images.
        return len(self.city_paths)
