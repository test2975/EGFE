import glob
from os import path
import argparse
from sketch_dataset.utils import get_missing_font

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract missing fonts from a logging folder')
    parser.add_argument('--folder', type=str, help='Folder of logfile to extract missing fonts from')
    args = parser.parse_args()
    font_missing_dict = {}
    for log in glob.glob(path.join(args.folder, "*.log")):
        for font, sketch_set in get_missing_font(log).items():
            font_missing_dict.setdefault(font, set()).update(sketch_set)
    for font in sorted(font_missing_dict.keys()):
        print(f"Missing font {font}:")
        for sketch in sorted(font_missing_dict[font]):
            print(f"\t{sketch}")
