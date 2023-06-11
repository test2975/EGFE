import json
from logging import config
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

LAYER_CLASS_MAP = {
    'symbolMaster': 0,
    'group': 1,
    'oval': 2,
    'polygon': 3,
    'rectangle': 4,
    'shapePath': 5,
    'star': 6,
    'triangle': 7,
    'shapeGroup': 8,
    'text': 9,
    'symbolInstance': 10,
    'slice': 11,
    'MSImmutableHotspotLayer': 12,
    'bitmap': 13,
}


class SketchDataset(Dataset):
    def __init__(self, index_json_path: str, data_folder: str, tokenizer: PreTrainedTokenizerBase, cache_dir: str = "./cache", use_cache: bool = False, lazy: bool = False, filter_text=True, use_fullimage=False):
        if filter_text:
            print("--- Will Remove Text From DataSet ---")
        self.filter_text = filter_text
        self.use_cache = use_cache
        self.use_fullimage = use_fullimage
        self.cache_dir = cache_dir
        self.data_folder = Path(data_folder)
        self.index_json_path = index_json_path
        self.index_json = json.load(open(index_json_path, 'r'))#[:40]
        self.img_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.artboard_detail: List[Dict[str, Any]] = []
        self.lazy = lazy
        self.norm_max = (-3000, 3000)
        self.data = self.load_data(tokenizer)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.lazy:
            artboard = self.index_json[idx]
            json_data = self.artboard_detail[idx]
            images = self.load_image(artboard)
            images = self.extract_tensor_from_assets(images, json_data)
            return (images, *self.data[idx])
        else:
            return self.data[idx]

    def load_data(self, tokenizer: PreTrainedTokenizerBase):
        if self.use_cache:
            os.makedirs(
                f"{self.cache_dir}/{ 'filter_text' if self.filter_text else 'origin'}", exist_ok=True)
            cache_path = f"{self.cache_dir}/{ 'filter_text' if self.filter_text else 'origin'}/{os.path.basename(self.index_json_path).replace('.json', '.pth')}"
            if os.path.exists(cache_path):
                print("Use cached data:", cache_path)
                return torch.load(cache_path)
        data = []
        text_count = 0
        filter_text_count = 0

        for artboard in tqdm(self.index_json, desc='Loading Artboards'):
            json_path = self.data_folder / artboard['json']
            json_data = json.load(open(json_path, 'r'))
            mask: torch.Tensor = None
            if self.filter_text:
                text_count += len(json_data['layers'])
                mask = torch.tensor([json_data['layers'][i]['_class'] !=
                                     'text' for i in range(len(json_data['layers']))])

                json_data['layers'] = list(filter(
                    lambda x: x['_class'] != 'text', json_data['layers']))
                filter_text_count += len(json_data['layers'])

            self.artboard_detail.append(json_data)
            if self.lazy:
                data.append(self.load_single_data(json_data, tokenizer, json_path))
            else:
                data.append((self.extract_tensor_from_assets(self.load_image(artboard), json_data, mask=mask),
                             *self.load_single_data(json_data, tokenizer, json_path)))
        if self.filter_text:
            print('Before filter text:', text_count)
            print('After filter text:', filter_text_count)
        if self.use_cache:
            torch.save(data, cache_path)
        return data

    def extract_tensor_from_assets(
            self,
            assets_image: Image,
            artboard_detail: Dict[str, Any],
            mask: torch.Tensor = None
    ) -> torch.Tensor:
        if self.use_fullimage:
            return self.img_transform(assets_image.convert("RGB"))
        single_layer_size = (
            artboard_detail['layer_width'], artboard_detail['layer_height'])
        asset_image_tensor = self.img_transform(assets_image.convert("RGB"))
        images = torch.stack(asset_image_tensor.split(
            single_layer_size[1], dim=1))

        if mask is not None:
            images = images[mask == True]
        return images

    def load_image(
            self,
            artboard_index: Dict[str, str]
    ) -> Image:
        if self.use_fullimage:
            full_image_path = str(
            self.data_folder / artboard_index['image'])
            image = Image.open(full_image_path)
            return image.resize((750, 750))
        asset_image_path = str(
            self.data_folder / artboard_index['layerassets'])
        return Image.open(asset_image_path)

    def load_single_data(
            self,
            artboard_detail: Dict[str, Any],
            tokenizer: PreTrainedTokenizerBase, file_path
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        names = []
        bboxes = []
        colors = []
        classes = []
        labels = []
        for layer in artboard_detail['layers']:
            layer_name = layer['name'].lower()
            names.append(tokenizer.encode(
                layer_name,
                add_special_tokens=False,
                padding=PaddingStrategy.MAX_LENGTH,
                truncation=True,
            ))
            x1, y1, width, height = layer['rect']['x'], layer['rect']['y'], layer['rect']['width'], layer['rect'][
                'height']
            x2, y2 = x1 + width, y1 + height
            # normalize the bbox
            if self.norm_max:
                x1, y1, x2, y2 = (float(x1)-self.norm_max[0])/(self.norm_max[1]-self.norm_max[0]) * 100.0,\
                    (float(y1)-self.norm_max[0])/(self.norm_max[1]-self.norm_max[0]) * 100.0,\
                    (float(x2)-self.norm_max[0])/(self.norm_max[1]-self.norm_max[0]) * 100.0,\
                    (float(y2)-self.norm_max[0])/(self.norm_max[1]-self.norm_max[0]) * 100.0
            bboxes.append([x1, y1, x2, y2])
            colors.append([color / 255.0 for color in layer['color']])
            classes.append(LAYER_CLASS_MAP[layer['_class']])
            labels.append(layer["label"])
        names = torch.as_tensor(names, dtype=torch.int64)
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        colors = torch.as_tensor(colors, dtype=torch.float32)
        classes = torch.as_tensor(classes, dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        return (names, bboxes, colors, classes, labels)


__all__ = ['SketchDataset']