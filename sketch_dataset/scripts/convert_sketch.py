import argparse
import glob
from dataclasses import dataclass
from os import path

from sketch_dataset.datasets import convert, ConvertedSketchConfig
from sketch_dataset.utils import create_folder

parser = argparse.ArgumentParser(
    description='Convert sketches in folder to dataset format',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--input', type=str, help='Folder containing sketches')
parser.add_argument('--output', type=str, help='Output folder')
parser.add_argument('--sketch_folder', type=str, help='Folder containing sketch folders', default='sketches')
parser.add_argument('--sketch_json', type=str, help='Name of sketch json', default="main.json")
parser.add_argument('--shrunk_sketch', type=str, help='Name of shrunk sketch file', default="main.sketch")
parser.add_argument('--artboard_json', type=str, help='Name of artboard json file contains layers bbox structure',
                    default="main.json")
parser.add_argument('--artboard_image', type=str, help='Name of exported artboard image file', default="main.png")
parser.add_argument('--config_file', type=str, help='File store sketch_name, artboard_json, artboard_image',
                    default="config.json")
parser.add_argument('--logfile_folder', type=str, help='Folder to save log files', default="{output}/logging")
parser.add_argument('--profile_folder', type=str, help='Folder to save profile files', default="{output}/profile")
parser.add_argument('--threads', type=int, help='Number of threads to use', default=8)


@dataclass
class ConvertNameSpace:
    input: str = ""
    output: str = ""
    sketch_folder: str = parser.get_default('sketch_folder')
    sketch_json: str = parser.get_default('sketch_json')
    shrunk_sketch: str = parser.get_default('shrunk_sketch')
    artboard_json: str = parser.get_default('artboard_json')
    artboard_image: str = parser.get_default('artboard_image')
    config_file: str = parser.get_default('config_file')
    logfile_folder: str = ""
    profile_folder: str = ""
    threads: int = parser.get_default('threads')


if __name__ == "__main__":
    args = parser.parse_args(namespace=ConvertNameSpace())

    if not path.isdir(args.input):
        raise ValueError("Input folder is not specified")
    if not args.output:
        raise ValueError("Output folder is not specified")
    if not args.logfile_folder:
        args.logfile_folder = path.join(args.output, "logging")
    if not args.profile_folder:
        args.profile_folder = path.join(args.output, "profile")
    # test_sketch = ['/Users/clq2021/Desktop/cyn-workspace/data/dataset_Ali/Alibaba_Sketch_4.sketch']
    for folder in [args.output, args.logfile_folder, args.profile_folder]:
        create_folder(folder)
    convert(
        # glob.glob(path.join(args.input, "*.sketch")),
        glob.glob(path.join(args.input, "*.sketch"))[:20],
        # test_sketch,
        ConvertedSketchConfig(
            args.sketch_json,
            args.shrunk_sketch,
            args.artboard_json,
            args.artboard_image,
            path.join(args.output, args.sketch_folder, args.config_file),
            path.join(args.output, args.sketch_folder)
        ),
        args.logfile_folder,
        args.profile_folder,
        args.threads
    )
