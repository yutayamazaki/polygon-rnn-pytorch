import glob
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms


def load_cityscapes(
    img_root: str = '../leftImg8bit_trainvaltest/',
    anno_root: str = '../gtFine_trainvaltest/'
) -> Dict[str, List[Tuple[str, str]]]:
    # train: 2975, val: 500, test: 1525
    datasets: Dict[str, List[Tuple[str, str]]] = {'train': [], 'val': [], 'test': []}
    for phase in ('train', 'val', 'test'):
        json_paths: List[str] = glob.glob(os.path.join(anno_root, 'gtFine', phase, '*', '*.json'))
        for json_path in json_paths:
            # file_id = 'munich_000187_000019_gtFine_'
            file_id: str = os.path.basename(json_path)[:-21]
            cityname: str = file_id.split('_')[0]

            img_path = os.path.join(img_root, 'leftImg8bit', phase, cityname, f'{file_id}_leftImg8bit.png')

            if os.path.exists(img_path) and os.path.exists(json_path):
                datasets[phase].append((img_path, json_path))

        assert len(json_paths) == len(datasets[phase])
    return datasets


class PolygonCityScapesDataset(torch.utils.data.dataset.Dataset):

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

    def _resize_polygon(self, polygon: List[List[int]]) -> List[List[float]]:
        polygon: np.ndarray = np.array(polygon)

        min_x: int = int(np.min(polygon[:, 0]))
        min_y: int = int(np.min(polygon[:, 1]))
        max_x: int = int(np.max(polygon[:, 0]))
        max_y: int = int(np.max(polygon[:, 1]))

        new_polygon: List[List[int]] = []
        for points in polygon:
            object_h: int = max_y - min_y
            object_w: int = max_x - min_x
            scale_h: float = 224.0 / object_h
            scale_w: float = 224.0 / object_w
            index_a: float = (points[0] - min_x) * scale_w
            index_b: float = (points[1] - min_y) * scale_h
            index_a: float = np.maximum(0, np.minimum(223, index_a))
            index_b: float = np.maximum(0, np.minimum(223, index_b))
            points: List[float] = [index_a, index_b]
            new_polygon.append(points)
        return new_polygon

    def __getitem__(self, index):
        # mock
        # img = torch.zeros((3, 224, 224))  # (B, C, H, W)
        # first = torch.zeros((28*28+3))
        # second = torch.zeros((58, 28*28+3))
        # third = torch.zeros((58, 28*28+3))
        # gt = torch.zeros([self.seq_len])[2:]
        # return (img, first, second, third, gt)

        img_path, json_path = self.city_paths[index]

        with open(json_path, 'r') as f:
            json_file: Dict[str, Any] = json.load(f)

        for obj in json_file['objects']:
            class_name: str = obj['label']
            polygon: List[List[int]] = obj['polygon']
            if class_name in self.class_names:
                break

        polygon_array: np.ndarray = np.array(polygon)
        min_x = np.min(polygon_array[:, 0])
        min_y = np.min(polygon_array[:, 1])
        max_x = np.max(polygon_array[:, 0])
        max_y = np.max(polygon_array[:, 1])

        img: Image.Image = Image.open(img_path).convert('RGB').crop((min_x, min_y, max_x, max_y))
        img: Image.Image = img.resize((224, 224), Image.BILINEAR)

        point_num: int = len(polygon)
        point_count: int = 2
        label_array: np.ndarray = np.zeros([self.seq_len, 28 * 28 + 3])
        label_index_array: np.ndarray = np.zeros([self.seq_len])
        if point_num < self.seq_len - 3:
            polygon: List[List[float]] = self._resize_polygon(polygon)
            polygon = np.array(polygon)
            for points in polygon:
                index_a: int = int(points[0] / 8)
                index_b: int = int(points[1] / 8)
                index: int = index_b * 28 + index_a
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
                    index_a: int = int(polygon[kkk % (point_num + 3) - 2][0] / 8)
                    index_b: int = int(polygon[kkk % (point_num + 3) - 2][1] / 8)
                    index: int = index_b * 28 + index_a
                label_array[kkk, index] = 1
                label_index_array[kkk] = index

        else:
            scale: float = point_num * 1.0 / (self.seq_len - 3)
            index_list = (np.arange(0, self.seq_len - 3) * scale).astype(int)
            polygon: List[List[float]] = self._resize_polygon(polygon)
            polygon = np.array(polygon)
            for points in polygon[index_list]:
                index_a: int = int(points[0] / 8)
                index_b: int = int(points[1] / 8)
                index: int = index_b * 28 + index_a
                label_array[point_count, index] = 1
                label_index_array[point_count] = index
                point_count += 1
            for kkk in range(point_count, self.seq_len):
                index: int = 28 * 28
                label_array[kkk, index] = 1
                label_index_array[kkk] = index

        img: torch.Tensor = self.transform(img)
        return (
            img, label_array[2], label_array[:-2],
            label_array[1:-1], label_index_array[2:]
        )

    def __len__(self) -> int:
        # Number of images.
        return len(self.city_paths)


def load_data(data_size: int, data_type: str, seq_len: int, batch_size: int):
    """
    Args:
        data_size (int): Number of images in this dataset.
        data_type (str): Specify 'train', 'val', 'test'.
        seq_len (int): Sequantial length of LSTM.
        batch_size (int): bacth size.
    Returns:
        torch.utils.data.DataLoader
    """
    transform = transforms.ToTensor()
    dataset: PolygonDataset = PolygonCityScapesDataset(
        data_size, data_type, seq_len, transform
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, drop_last=False
    )
    return data_loader


if __name__ == '__main__':
    batch_size: int = 3
 
    datasets = load_cityscapes()
    for phase in ('train', 'val', 'test'):
        print(len(datasets[phase]))

    dtrain = PolygonCityScapesDataset(city_paths=datasets['train'], transform=transforms.ToTensor())
    dval = PolygonCityScapesDataset(city_paths=datasets['val'], transform=transforms.ToTensor())
    dtest = PolygonCityScapesDataset(city_paths=datasets['test'], transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(
        dtrain, batch_size=batch_size,
        shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(dtrain, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(dtrain, batch_size=batch_size)

    for step, data in enumerate(train_loader):
        x = Variable(data[0])
        x1 = Variable(data[1])
        x2 = Variable(data[2])
        x3 = Variable(data[3])
        ta = Variable(data[4])

        print(x.size(), x1.size(), x2.size(), x3.size(), ta.size())
