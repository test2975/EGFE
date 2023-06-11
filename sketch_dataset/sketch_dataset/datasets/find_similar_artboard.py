# encoding: utf-8
import glob
import logging
from functools import cache
from os import path
from pathlib import Path
from queue import Queue
from typing import List, Dict, Tuple, Set

import numpy as np
from PIL import Image
from tqdm import tqdm

from sketch_dataset.utils import get_png_size, ProfileLoggingThread

Image.MAX_IMAGE_PIXELS = None
MIN_WIDTH = 200
MAX_WIDTH = 2000
MIN_HEIGHT = 200
MAX_HEIGHT = 20000


@cache
def get_image(image_path: str) -> np.ndarray:
    return np.asarray(Image.open(image_path).convert("RGB"))


def compare_image(image1: np.ndarray, image2: np.ndarray) -> bool:
    x = min(image1.shape[0], image2.shape[0])
    y = min(image1.shape[1], image2.shape[1])
    c = min(image1.shape[2], image2.shape[2])

    res = np.mean((image1[:x, :y, :c] - image2[:x, :y, :c]) ** 2)
    return res < 10


class MergeArtboardGroupThread(ProfileLoggingThread):
    def __init__(
            self,
            input_artboard_list_queue: Queue[Tuple[List[str], Tuple[int, int]]],
            output_groups_queue: Queue[List[str]],
            thread_name: str,
            logging_file: str,
            profile_file: str,
            pbar: tqdm
    ):
        super().__init__(
            thread_name,
            logging_file,
            profile_file
        )
        self.input_artboard_list_queue = input_artboard_list_queue
        self.output_groups_queue = output_groups_queue
        self.pbar = pbar

    def run_impl(self):
        while True:
            if self.input_artboard_list_queue.empty():
                break
            try:
                artboard_list, size = self.input_artboard_list_queue.get()
                sub_image_groups = []
                self.logger.info(f'start processing {len(artboard_list)} artboards with size {size}')
                for artboard in artboard_list:
                    image = get_image(artboard)
                    match = False
                    compare_count = 0
                    for group in sub_image_groups:
                        compare_count += 1
                        if compare_image(image, group[0]):
                            group.append(artboard)
                            group[0] = ((group[0] * len(group) + image) / len(group)).astype(np.uint8)
                            match = True
                            break
                    if not match:
                        sub_image_groups.append([image, artboard])
                    self.logger.info(f'processed {artboard}, compared {compare_count} images')
                    self.pbar.update(1)
                for group in sub_image_groups:
                    self.output_groups_queue.put(group[1:])
            except Exception as e:
                self.logger.error(e)
            finally:
                self.input_artboard_list_queue.task_done()


def merge_artboard_group(
        sketch_folders: List[str],
        logging_folder: str,
        profile_folder: str,
        max_thread=8
) -> List[List[str]]:
    logging.info(f"{len(sketch_folders)} sketches found")
    artboards: List[Tuple[str, int, int]] = []
    for sketch_folder in tqdm(sketch_folders):
        for artboard_path in glob.glob(f"{sketch_folder}/*"):
            image_size = get_png_size(artboard_path)
            artboards.append((artboard_path, image_size[0], image_size[1]))
    logging.info(f"{len(artboards)} artboards found")
    size_groups: Dict[Tuple[int, int], List[str]] = {}
    for artboard_path, width, height in artboards:
        size_groups.setdefault((width, height), []).append(artboard_path)
    logging.info(f"{len(size_groups)} size groups found")
    size_groups = {
        key: value
        for key, value in size_groups.items()
        if len(value) > 1
    }
    logging.info(f"{len(size_groups)} size groups with more than 1 artboard found")
    size_groups = {
        key: value
        for key, value in size_groups.items()
        if MIN_WIDTH < key[0] <= MAX_WIDTH and MIN_HEIGHT < key[1] <= MAX_HEIGHT
    }
    logging.info(f"{len(size_groups)} size groups with width between 100 and 1000 found")
    image_group_queue = Queue()
    input_artboard_queue = Queue()
    pbar = tqdm(total=sum([len(x) for x in size_groups.values()]))
    for size, image_paths in size_groups.items():
        input_artboard_queue.put((image_paths, size))
    for i in range(max_thread):
        thread_name = f"simgroup-thread-{i}"
        thread = MergeArtboardGroupThread(
            input_artboard_queue,
            image_group_queue,
            thread_name,
            path.join(logging_folder, f"{thread_name}.log"),
            path.join(profile_folder, f"{thread_name}.profile"),
            pbar
        )
        thread.daemon = True
        thread.start()
    input_artboard_queue.join()
    return [image_group for image_group in image_group_queue.queue if len(image_group) > 1]


def visualize_groups(image_groups: List[List[str]], output_folder: str):
    for group_idx, group in enumerate(tqdm(image_groups)):
        group_images = [Image.open(x) for x in group]
        new_image = Image.new(
            mode='RGB',
            size=(group_images[0].size[0] * len(group_images), group_images[0].size[1])
        )
        for i, image in enumerate(group_images):
            new_image.paste(image, (i * image.size[0], 0))
        new_image.save(f"{output_folder}/{group_idx}.png")


def find_similar_sketch(
        image_groups: List[List[str]]
):
    adjacency_list = {}
    for group in image_groups:
        first_sketch = Path(group[0]).parent.name
        entry = adjacency_list.setdefault(first_sketch, set())
        for image_path in group[1:]:
            sketch = Path(image_path).parent.name
            if sketch != first_sketch:
                adjacency_list.setdefault(sketch, set()).add(first_sketch)
                entry.add(sketch)
    groups = []

    def recursive_find(sketch: str, group: Set[str]):
        neighbors = adjacency_list.get(sketch, set())
        while len(neighbors) > 0:
            neighbor = neighbors.pop()
            group.add(neighbor)
            recursive_find(neighbor, group)

    for sketch in adjacency_list:
        group_set = set()
        group_set.add(sketch)
        recursive_find(sketch, group_set)
        groups.append(list(sorted(group_set)))
    return list(sorted(groups, key=lambda x: len(x)))


__all__ = [
    "merge_artboard_group",
    "visualize_groups",
    "find_similar_sketch"
]
