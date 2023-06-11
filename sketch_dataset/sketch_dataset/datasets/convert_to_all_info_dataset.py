import asyncio
import logging
import traceback
from dataclasses import dataclass
from functools import reduce
from os import path, makedirs
from pathlib import Path
from queue import Queue
from shutil import move, rmtree
from typing import List, Dict, Any, Generic, TypeVar, Type

from fastclasses_json import dataclass_json, JSONMixin
from sketch_document_py.sketch_file import from_file, to_file, SketchFile
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from sketch_dataset.datasets.extend_sketch_file_format import ExtendArtboard
from sketch_dataset.sketchtool import SketchToolWrapper, DEFAULT_SKETCH_PATH, ExportFormat, ListLayer
from sketch_dataset.utils import extract_artboards_from_sketch, ProfileLoggingThread

ITEM_THRESHOLD = 1000


@dataclass_json
@dataclass
class ConvertedSketchConfig(JSONMixin):
    sketch_json: str
    sketch_file: str
    artboard_json: str
    artboard_image: str
    config_file: str
    output_dir: str
    sketches: List[str] = ()


@dataclass_json
@dataclass
class ConvertedSketch(JSONMixin):
    artboards: List[str] = ()


T = TypeVar('T', bound='ListLayer')


@dataclass
class ArtboardData(Generic[T]):
    sketch_folder: 'Path'
    artboard_folder: 'Path'
    list_layer: 'T'
    main_image: 'Path'
    layer_images: Dict[str, 'Path']


@dataclass
class SketchData(Generic[T]):
    sketch_folder: 'Path'
    sketch_path: 'Path'
    artboards: List['ArtboardData[T]']


@dataclass
class Dataset(Generic[T]):
    config: 'ConvertedSketchConfig'
    sketches: List['SketchData[T]']

    @classmethod
    def from_config(cls, config_path: str, list_layer_type: Type[T]) -> 'Dataset[T]':
        config_parent = Path(config_path).parent
        config: ConvertedSketchConfig = ConvertedSketchConfig.from_json(open(config_path).read())
        config.output_dir = str(config_parent)

        def read_sketch(sketch: str) -> SketchData:
            sketch_folder = config_parent.joinpath(sketch)
            sketch_path = sketch_folder.joinpath(config.sketch_file)
            artboards = []
            sketch_json: ConvertedSketch = ConvertedSketch.from_json(
                open(sketch_folder.joinpath(config.sketch_json)).read())
            for artboard in sketch_json.artboards:
                artboard_folder = sketch_folder.joinpath(artboard)
                list_layer: T = list_layer_type.from_json(
                    open(artboard_folder.joinpath(config.artboard_json)).read())
                flatten_layer = list_layer.flatten()
                artboards.append(ArtboardData(
                    sketch_folder=sketch_folder,
                    artboard_folder=artboard_folder,
                    list_layer=list_layer,
                    main_image=artboard_folder.joinpath(config.artboard_image),
                    layer_images={
                        layer.id: artboard_folder.joinpath(f"{layer.id}.png")
                        for layer in flatten_layer
                    }
                ))
            return SketchData(
                sketch_folder=sketch_folder,
                sketch_path=sketch_path,
                artboards=artboards
            )

        sketches = thread_map(read_sketch, config.sketches, max_workers=12, desc="Reading Sketch")
        return cls(config=config, sketches=sketches)


def recursive_merge(dict1: Dict[str, Any], dict2: Dict[str, Any], keys: List[str], strict: bool = False):
    result = {**dict1, **dict2}
    for key in keys:
        if key in dict1 and key in dict2:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                result[key] = recursive_merge(dict1[key], dict2[key], keys, strict)
            elif isinstance(dict1[key], list) and isinstance(dict2[key], list):
                new_value = []
                if strict:
                    assert len(dict1[key]) == len(dict2[key])
                for value1, value2 in zip(dict1[key], dict2[key]):
                    if isinstance(value1, dict) and isinstance(value2, dict):
                        new_value.append(recursive_merge(value1, value2, keys, strict))
                    else:
                        new_value.append((value1, value2))
                result[key] = new_value
    return result


async def convert_sketch(
        sketch_path: str,
        convert_config: ConvertedSketchConfig,
        logger: logging.Logger = None,
        strict: bool = True
) -> str:
    """
    Convert a sketch file to dataset.

    e.g.
    a sketch contains 3 artboards:
    1. artboard_1 (id: id_1) contains 1 layers (id: id_1_1, )
    2. artboard_2 (id: id_2) contains 2 layers (id: id_2_1, id: id_2_2)
    3. artboard_3 (id: id_3) contains 3 layers (id: id_3_1, id: id_3_2, id: id_3_3)

    resulted file tree will be:
    output_folder/
        sketch_name/
            id_1/
                id_1_1.png
                artboard_json_name.json
                artboard_export_image_name.png
            id_2/
                id_2_1.png
                id_2_2.png
                artboard_json_name.json
                artboard_export_image_name.png
            id_3/
                id_3_1.png
                id_3_2.png
                id_3_3.png
                artboard_json_name.json
                artboard_export_image_name.png
            sketch_json_name.json
            sketch_file_name.sketch

    """
    sketch_output_folder = path.join(convert_config.output_dir, Path(sketch_path).stem)
    path.isdir(sketch_output_folder) or makedirs(sketch_output_folder, exist_ok=True)
    shrink_sketch_path = path.join(sketch_output_folder, convert_config.sketch_file)

    try:
        # save shrunk sketch
        sketch_file: SketchFile
        sketch_file = from_file(sketch_path)
        to_file(sketch_file, shrink_sketch_path)

        # init sketchtool
        sketchtool = SketchToolWrapper(DEFAULT_SKETCH_PATH)

        # read list layers
        list_layers_coroutine = sketchtool.list.layers(sketch_path)
        list_layers_coroutine_res = await list_layers_coroutine
        if list_layers_coroutine_res.stderr and logger:
            logger.warning(f'convert {sketch_path} failed when list layers: ')
            logger.warning(list_layers_coroutine_res.stderr)
        first_level_layer_dict: Dict[str, ListLayer] = {
            layer.id: layer
            for page in list_layers_coroutine_res.value.pages
            for layer in page.layers
        }

        # export artboard main image
        artboard_dict: Dict[str, ExtendArtboard] = {
            artboard.do_objectID: ExtendArtboard.from_dict(
                recursive_merge(
                    artboard.to_dict(),
                    first_level_layer_dict[artboard.do_objectID].to_dict(),
                    ['layers']
                )
            )
            for artboard in extract_artboards_from_sketch(sketch_file)
        }

        export_format = ExportFormat.PNG
        export_artboards_coroutine = sketchtool.export.artboards(
            sketch_path,
            output=sketch_output_folder,
            formats=[export_format],
            items=artboard_dict.keys()
        )
        export_artboards_coroutine_res = await export_artboards_coroutine
        if export_artboards_coroutine_res.stderr and logger:
            logger.warning(f'convert {sketch_path} failed when export artboards: ')
            logger.warning(export_artboards_coroutine_res.stderr)
        export_result: Dict[str, str] = {k: v[0] for k, v in export_artboards_coroutine_res.value.items()}

        export_layers_coroutines = []
        for artboard_id, artboard_export_path in export_result.items():
            if artboard_export_path:
                # make artboard dir
                artboard_output_folder = path.join(sketch_output_folder, artboard_id)
                path.isdir(artboard_output_folder) or makedirs(artboard_output_folder, exist_ok=True)

                # get artboard json
                artboard_json = artboard_dict[artboard_id]

                # get all layer id in artboard
                items = [layer.id for layer in artboard_json.flatten()]

                # export artboard layers
                for i in range(int(len(items) / ITEM_THRESHOLD) + 1):
                    export_layers_coroutine = sketchtool.export.layers(
                        sketch_path,
                        output=artboard_output_folder,
                        formats=[export_format],
                        items=items[i * ITEM_THRESHOLD: (i + 1) * ITEM_THRESHOLD],
                    )
                    export_layers_coroutines.append(export_layers_coroutine)

                # write artboard json
                artboard_json_path = path.join(artboard_output_folder, convert_config.artboard_json)
                with open(artboard_json_path, "w") as layer_json:
                    layer_json.write(artboard_json.to_json())

                # write artboard image
                artboard_image_path = path.join(artboard_output_folder, convert_config.artboard_image)
                move(artboard_export_path, artboard_image_path)
            else:
                if strict:
                    raise Exception(f'export artboard {artboard_id} failed')
        for export_layers_coroutine in export_layers_coroutines:
            export_layers_coroutine_res = await export_layers_coroutine
            if export_layers_coroutine_res.stderr and logger:
                logger.warning(f'convert {sketch_path} failed when export layers: ')
                logger.warning(export_layers_coroutine_res.stderr)
            if strict:
                if not reduce(
                        lambda x, y: x and y,
                        export_layers_coroutine_res.value.values(),
                        True
                ):
                    raise Exception(f'export layers failed')
        with open(path.join(sketch_output_folder, convert_config.sketch_json), "w") as sketch_json:
            sketch_json.write(ConvertedSketch(list(export_result.keys())).to_json())
    except Exception as e:
        rmtree(sketch_output_folder)
        raise e
    return Path(sketch_path).stem


def convert_sketch_sync(
        sketch_path: str,
        convert_config: ConvertedSketchConfig,
        logger: logging.Logger = None
) -> str:
    return asyncio.run(
        convert_sketch(sketch_path, convert_config, logger))


class ConvertSketchThread(ProfileLoggingThread):
    def __init__(
            self,
            sketch_queue: Queue[str],
            result_queue: Queue[str],
            convert_config: ConvertedSketchConfig,
            thread_name: str,
            logfile_path: str,
            profile_path: str,
            pbar: tqdm,
    ):
        super().__init__(thread_name, logfile_path, profile_path)
        self.sketch_queue = sketch_queue
        self.result_queue = result_queue
        self.convert_config = convert_config
        self.pbar = pbar

    def run_impl(self):
        while True:
            if self.sketch_queue.empty():
                break
            sketch_path = self.sketch_queue.get()
            try:
                self.result_queue.put(convert_sketch_sync(
                    sketch_path,
                    self.convert_config,
                    self.logger
                ))
            except Exception as e:
                self.logger.error(f"encounter exception when converting {sketch_path}: \n{traceback.format_exc()}")
            finally:
                self.pbar.update()
                self.sketch_queue.task_done()


def convert(
        sketch_list: List[str],
        convert_config: ConvertedSketchConfig,
        logfile_folder: str,
        profile_folder: str,
        max_threads: int = 4,
):
    pbar = tqdm(total=len(sketch_list))
    sketch_queue: Queue[str] = Queue()
    result_queue: Queue[str] = Queue()
    for sketch_path in sketch_list:
        sketch_queue.put(sketch_path)
    for i in range(max_threads):
        thread_name = f"convert-thread-{i}"
        thread = ConvertSketchThread(
            sketch_queue,
            result_queue,
            convert_config,
            thread_name,
            path.join(logfile_folder, f"{thread_name}.log"),
            path.join(profile_folder, f"{thread_name}.profile"),
            pbar
        )
        thread.daemon = True
        thread.start()
    sketch_queue.join()
    convert_config.output_dir = "."
    convert_config.sketches = tuple(result_queue.queue)
    with open(path.join(convert_config.output_dir, convert_config.config_file), "w") as f:
        f.write(convert_config.to_json())


__all__ = ['convert', 'ConvertedSketchConfig', 'Dataset', 'ArtboardData']
