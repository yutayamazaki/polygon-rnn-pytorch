import copy
import glob
import json
import math
import os
import random
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


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


def sort_polygon(polygon: List[List[int]]) -> List[List[int]]:
    copied_polygon: List[List[int]] = list(set(polygon))
    # copied_polygon: List[List[int]] = copy.deepcopy(polygon)
    sorted_polygon: List[List[int]] = []

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


def get_size(polygon: List[List[int]]) -> Tuple[int, ...]:
    polygon_arr: np.ndarray = np.array(polygon)
    x_max: int = int(np.max(polygon_arr[:, 0]))
    x_min: int = int(np.min(polygon_arr[:, 0]))
    y_max: int = int(np.max(polygon_arr[:, 1]))
    y_min: int = int(np.min(polygon_arr[:, 1]))
    height: int = y_max - y_min
    width: int = x_max - x_min

    return (x_min, x_max, y_min, y_max, height, width)


def crop_object(
    img: Image.Image, polygon: List[List[int]],
    size: Tuple[int, int] = (224, 224)
) -> Tuple[Image.Image, List[List[int]]]:
    x_min, x_max, y_min, y_max, obj_h, obj_w = get_size(polygon)

    shape: Tuple[int, ...] = (x_min, y_min, x_max, y_max)
    img_crop_resized: Image.Image = img.crop(shape).resize(size)

    polygon_resized: List[List[int]] = resize_polygon(polygon)

    return img_crop_resized, polygon_resized


def resize_polygon(
    polygon: List[List[int]], size: Tuple[int, int] = (224, 224)
) -> List[List[int]]:
    x_min, x_max, y_min, y_max, obj_h, obj_w = get_size(polygon)
    crop_w: int = size[0]
    crop_h: int = size[1]

    scale_h: float = crop_h / obj_h
    scale_w: float = crop_w / obj_w

    resized: List[List[int]] = []
    for p in polygon:
        w: int = int(min(max((p[0] - x_min) * scale_w, 0), crop_w - 1))
        h: int = int(min(max((p[1] - y_min) * scale_h, 0), crop_h - 1))
        resized.append([w, h])
    return resized


if __name__ == '__main__':
    from datasets import PolygonCityScapesDataset
    from models import PolygonRNN
    from train import load_yaml
    import utils

    utils.seed_everything()

    crop_h: int = 224
    crop_w: int = 224

    images: List[str] = glob.glob('../leftImg8bit_trainvaltest/leftImg8bit/train/*/*.png')
    for img_path in images[:10]:
        fname: str = os.path.basename(img_path)
        file_id, _ = os.path.splitext(fname)
        file_id = file_id[:-12]
        city: str = file_id.split('_')[0]
        json_path: str = f'../gtFine_trainvaltest/gtFine/train/{city}/{file_id}_gtFine_polygons.json'

        with open(json_path, 'r') as f:
            anno = json.load(f)

        while True:
            obj = random.choice(anno['objects'])
            # obj: Dict[str, Any] = random.choice(objects)
            if obj['label'] in ['bike', 'bus', 'car', 'person', 'mbike', 'truck', 'rider', 'train']:
                break
            polygon: List[List[int]] = random.choice(obj['polygon'])

            # Calc height and width of object.

        class_name: str = obj['label']
        polygon: List[List[int]] = obj['polygon']
        polygon = sort_polygon(polygon)

        x_min, x_max, y_min, y_max, obj_h, obj_w = get_size(polygon)
        img_h: int = anno['imgHeight']
        img_w: int = anno['imgWidth']

        # Save prediction result
        fig = plt.figure()
        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2)
        ax3 = fig.add_subplot(3, 1, 3)

        orig_image: Image.Image = Image.open(img_path).convert('RGB')
        ax1.imshow(orig_image)
        ax1.set_title('Original')

        orig2: Image.Image = Image.open(img_path).convert('RGB')
        draw = ImageDraw.Draw(orig2)
        draw.polygon(xy=[(p[0], p[1]) for p in polygon], outline=(255, 0, 0))
        ax2.imshow(orig2)
        ax2.set_title('Annotation')

        orig3: Image.Image = Image.open(img_path).convert('RGB')
        orig3, polygon_resized = crop_object(orig3, polygon)

        draw = ImageDraw.Draw(orig3)
        draw.polygon(xy=[(p[0], p[1]) for p in polygon_resized], outline=(255, 0, 0))

        ax3.imshow(orig3)
        ax3.set_title('Scaled')

        plt.tight_layout()
        fig.savefig(f'./demo/hoge_{file_id}.png')
        plt.clf()

        print(obj_w, obj_h)

        del fig, ax1, ax2
