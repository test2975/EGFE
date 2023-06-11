from os import path
from shutil import move
from pathlib import Path

from sketch_dataset.datasets import ConvertedSketchConfig

config_path = "C:\\nozomisharediskc\\converted\\sketches\\config.json"
config: ConvertedSketchConfig = ConvertedSketchConfig.from_json(open(config_path, "r").read())
base_path = path.join(Path(config_path).parent, config.output_dir)
sketch_abs_paths = []
for sketch_folder in config.sketches:
    sketch_folder_path = path.join(base_path, sketch_folder)
    if not path.exists(sketch_folder_path):
        raise Exception(f"{sketch_folder_path} does not exist")
    sketch_abs_paths.append(sketch_folder_path)
for index, sketch_folder in enumerate(sketch_abs_paths):
    move(sketch_folder, path.join(base_path, f"{index:03d}"))
config.sketches = [f"{index:03d}" for index in range(len(sketch_abs_paths))]
open(config_path, "w").write(config.to_json())
print("Done")
