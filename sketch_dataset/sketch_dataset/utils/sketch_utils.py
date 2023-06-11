import re
from typing import List, Dict, Set

from sketch_document_py import sketch_file as sf
from sketch_document_py import sketch_file_format as sff


def flatten_sketch_group(sketch_group: sff.AnyGroup) -> List[sff.AnyLayer]:
    """
    Flatten a SketchGroup object into a list of Sketch objects.
    """
    layers = []
    group_layers = sketch_group.layers
    sketch_group.layers = []
    layers.append(sketch_group)
    for layer in group_layers:
        if isinstance(layer, sff.AnyGroup.__args__):
            layers.extend(flatten_sketch_group(layer))
        else:
            layers.append(layer)
    return layers


def extract_artboards_from_sketch(sketch_file: sf.SketchFile) -> List[sff.Artboard]:
    """
    Flatten a SketchGroup object into a list of Sketch objects.
    """
    pages = sketch_file.contents.document.pages
    return [
        layer
        for page in pages
        for layer in page.layers
        if isinstance(layer, sff.Artboard)
    ]


def get_missing_font(error_file: str) -> Dict[str, Set[str]]:
    missing_fonts = {}
    with open(error_file, "r") as f:
        lines = f.readlines()
        current_sketch_file = None
        for line in lines:
            sketch_name = re.match(r".+convert (.+\.sketch)", line)
            if sketch_name:
                current_sketch_file = sketch_name.group(1)
                continue
            missing_font_match = re.match(r".+Client requested name \"(((?!\").)+)\".+", line)
            if missing_font_match:
                missing_fonts.setdefault(missing_font_match.group(1), set()).add(current_sketch_file)
    return missing_fonts
