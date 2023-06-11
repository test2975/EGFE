from transformers import PreTrainedTokenizer

from .dataset import *


def build_dataset(index_path: str, data_folder: str, tokenizer: PreTrainedTokenizer, cache_dir: str = "./cache", use_cache: bool = False, remove_text: bool = True, use_fullimage: bool = False) -> SketchDataset:
    return SketchDataset(index_path, data_folder, tokenizer, cache_dir=cache_dir, use_cache=use_cache, filter_text=remove_text, use_fullimage=use_fullimage)
