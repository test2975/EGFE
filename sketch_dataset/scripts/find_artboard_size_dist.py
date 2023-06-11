import glob
import json
import sys
from os import path
from typing import List

import numpy as np
from PIL import Image

from sketch_dataset.datasets import BaseListLayer


def recover_artboard_from_asset(folder: str, index: str):
    data = json.loads(open(path.join(folder, f"{index}.json"), "r").read())
    layer_list: List[BaseListLayer] = [BaseListLayer.from_dict(layer) for layer in data["layers"]]
    assets_image = Image.open(path.join(folder, f"{index}-assets.png"))
    assets_list = [assets_image.crop((0, index * 64, 64, (index + 1) * 64)).resize(
        (int(layer.rect.width), int(layer.rect.height))) for index, layer in
        enumerate(layer_list)]
    original_image = Image.open(path.join(folder, f"{index}.png"))
    canvas = Image.new("RGBA", (int(data["width"]), int(data["height"])))
    for layer, image in zip(layer_list, assets_list):
        canvas.alpha_composite(image, (
            int(layer.rect.x), int(layer.rect.y)))
    compare = Image.new("RGBA", (canvas.width * 2, canvas.height), (255, 255, 255, 255))
    compare.paste(canvas, (0, 0))
    compare.paste(original_image, (canvas.width, 0))
    compare.show()


def calculate_artboard_size_distribution(folder: str):
    seq_len_dict = {}
    match_seq_len_artboards = set()
    name_len_dict = {}
    size_dict = {}
    match_size_artboards = set()
    all_json = set(glob.glob(path.join(folder, "*.json"))) - {path.join(folder, "index.json")}
    for json_file in all_json:
        info = json.loads(open(json_file, "r").read())
        seq_len = len(info["layers"])
        if 20 < seq_len < 200:
            match_seq_len_artboards.add(json_file)
        seq_len_dict[seq_len] = seq_len_dict.get(seq_len, 0) + 1
        image_size = (info["width"], info["height"])
        if image_size[0] < 2000:
            match_size_artboards.add(json_file)
        size_dict[image_size] = size_dict.get(image_size, 0) + 1
        for layer in info["layers"]:
            name_len = len(layer["name"])
            # if name_len <2:
            #     print(layer["name"])
            name_len_dict[name_len] = name_len_dict.get(name_len, 0) + 1
    print(f"match seq_len range artboard proportion: {len(match_seq_len_artboards)}/{len(all_json)}")
    print(sorted(seq_len_dict.items(), key=lambda x: x[0]))
    print(f"max sequence length: {max(seq_len_dict.keys())}")
    print(f"min sequence length: {min(seq_len_dict.keys())}")
    print(f"avg sequence length: {sum(seq_len_dict.keys()) / len(seq_len_dict.keys())}")
    print(f"match size range artboard proportion: {len(match_size_artboards)}/{len(all_json)}")
    print(sorted(size_dict.items(), key=lambda x: x[0]))
    print(f"max image size: {max(size_dict.keys())}")
    print(f"min image size: {min(size_dict.keys())}")
    print(f"avg image size: {np.sum(np.array(list(size_dict.keys())), axis=0) / len(size_dict.keys())}")
    print(
        f"match seq_len and range artboard proportion: {len(match_seq_len_artboards.intersection(match_size_artboards))}/{len(all_json)}")
    print(sorted(name_len_dict.items(), key=lambda x: x[0]))
    print(f"max name length: {max(name_len_dict.keys())}")
    print(f"min name length: {min(name_len_dict.keys())}")
    print(f"avg name length: {sum(name_len_dict.keys()) / len(name_len_dict.keys())}")


if __name__ == "__main__":
    # calculate_artboard_size_distribution(sys.argv[1])
    recover_artboard_from_asset(sys.argv[1], sys.argv[2])