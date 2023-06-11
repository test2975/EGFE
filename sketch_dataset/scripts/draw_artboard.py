from sketch_dataset.utils import draw_artboard
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input artboard folder")
    parser.add_argument("--artboard-json", type=str, default="main.json", help="Converted artboard json file")
    parser.add_argument("--artboard-image", type=str, default="main.png", help="Converted artboard image file")
    parser.add_argument("--output", type=str, help="Output image")
    args = parser.parse_args()
    draw_artboard(
        args.input,
        args.artboard_json,
        args.artboard_image,
        args.output
    )