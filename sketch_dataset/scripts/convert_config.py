from sketch_dataset.datasets import convert_from_config
from sketch_dataset.utils import create_folder
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert config to dataset')
    parser.add_argument('config', type=str,
                        help='Path to config file')
    parser.add_argument('output_dir', type=str,
                        help='Path to output directory')
    parser.add_argument('--size', type=int, default=64,
                        help='Size of the images')
    args = parser.parse_args()
    create_folder(args.output_dir)
    convert_from_config(args.config, args.output_dir, (args.size, args.size))
