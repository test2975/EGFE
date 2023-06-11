# encoding: utf-8
import asyncio
from os import path, makedirs
from pathlib import Path
from queue import Queue
from typing import List, Dict

from tqdm import tqdm

from sketch_dataset.sketchtool import SketchToolWrapper, DEFAULT_SKETCH_PATH, WrapperResult
from sketch_dataset.utils import ProfileLoggingThread


async def generate_artboards_png(sketch_path: str, output: str) -> WrapperResult[Dict[str, List[bool]]]:
    sketchtool = SketchToolWrapper(DEFAULT_SKETCH_PATH)
    artboards = [artboard for page in (await sketchtool.list.artboards(sketch_path)).value.pages for artboard in
                 page.artboards]
    output = path.join(output, Path(sketch_path).stem)
    path.isdir(output) or makedirs(output, exist_ok=True)
    return await sketchtool.export.artboards(
        sketch_path,
        output=output,
        items=[artboard.id for artboard in artboards],
        overwriting=True
    )


def generate_artboards_png_sync(sketch_path: str, output: str) -> WrapperResult[Dict[str, List[bool]]]:
    return asyncio.run(generate_artboards_png(sketch_path, output))


class GenerateArtboardsThread(ProfileLoggingThread):
    def __init__(
            self,
            sketch_queue: Queue,
            output: str,
            thread_name: str,
            logfile_path: str,
            profile_path: str,
            pbar: tqdm
    ):
        super().__init__(thread_name, logfile_path, profile_path)
        self.sketch_queue = sketch_queue
        self.output = output
        self.pbar = pbar

    def run_impl(self):
        while True:
            if self.sketch_queue.empty():
                break
            sketch_path = self.sketch_queue.get()
            self.logger.info(f'Generating artboards for {sketch_path}')
            res = generate_artboards_png_sync(sketch_path, self.output)
            self.logger.info(res.stdout)
            if res.stderr:
                self.logger.error(res.stderr)
            self.pbar.update()


def export_artboards(
        sketch_list: List[str],
        output_folder: str,
        logfile_folder: str,
        profile_folder: str,
        max_thread=8
):
    pbar = tqdm(total=len(sketch_list))
    sketch_queue = Queue()
    for sketch_path in sketch_list:
        sketch_queue.put(sketch_path)
    for i in range(max_thread):
        thread_name = f"export-thread-{i}"
        thread = GenerateArtboardsThread(
            sketch_queue,
            output_folder,
            thread_name,
            path.join(logfile_folder, f"{thread_name}.log"),
            path.join(profile_folder, f"{thread_name}.profile"),
            pbar
        )
        thread.daemon = True
        thread.start()
    sketch_queue.join()


__all__ = [
    "export_artboards"
]
