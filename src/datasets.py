import json

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class PolygonDataset(torch.utils.data.dataset.Dataset):

    def __init__(
        self,
        data_size: int,
        dataset: str,
        seq_len: int,
        transform=None
    ):
        self.data_size: int = data_size
        self.dataset: str = dataset
        self.seq_len: int = seq_len
        self.transform = transform

    def __getitem__(self, index):
        # mock
        img = torch.zeros((3, 224, 224))  # (B, C, H, W)
        first = torch.zeros((28*28+3))
        second = torch.zeros((58, 28*28+3))
        third = torch.zeros((58, 28*28+3))
        gt = torch.zeros([self.seq_len])[2:]
        return (img, first, second, third, gt)

        img_name = 'new_img/{}/{}.png'.format(self.dataset, index)
        label_name = 'new_label/{}/{}.json'.format(self.dataset, index)
        try:
            img = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            return None
        assert not (img is None)
        json_file: dict = json.load(open(label_name))
        point_num: int = len(json_file['polygon'])
        polygon: np.ndarray = np.array(json_file['polygon'])
        point_count: int = 2
        # img_array = np.zeros([data_num, 3, 224, 224])
        label_array: np.ndarray = np.zeros([self.seq_len, 28 * 28 + 3])
        label_index_array: np.ndarray = np.zeros([self.seq_len])
        if point_num < self.seq_len - 3:
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
                    index_a = int(polygon[kkk % (point_num + 3) - 2][0] / 8)
                    index_b = int(polygon[kkk % (point_num + 3) - 2][1] / 8)
                    index: int = index_b * 28 + index_a
                label_array[kkk, index] = 1
                label_index_array[kkk] = index
        else:
            scale: float = point_num * 1.0 / (self.seq_len - 3)
            index_list = (np.arange(0, self.seq_len - 3) * scale).astype(int)
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

        if self.transform is not None:
            img = self.transform(img)
        # stride = self.seq_len - 2
        return (img, label_array[2], label_array[:-2], label_array[1:-1],
                label_index_array[2:])

    def __len__(self) -> int:
        # Number of images.
        return self.data_size


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
    dataset: PolygonDataset = PolygonDataset(
        data_size, data_type, seq_len, transform
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, drop_last=False
    )
    return data_loader
