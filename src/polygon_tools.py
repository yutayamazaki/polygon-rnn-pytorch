import copy
import math
from typing import List, Tuple

import numpy as np
from PIL import Image


def index2point(
    index: int, height: int = 28, width: int = 28
) -> Tuple[int, int]:
    num_rows: int = index // width
    num_cols: int = index % width
    return (num_rows, num_cols)


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def sort_polygon(polygon: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    copied_polygon: List[Tuple[int, int]] = list(set(polygon))  # type: ignore
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
                # Need deepcopy?
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
