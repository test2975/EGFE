from os import path
from typing import Optional

from PIL import Image

from sketch_dataset.sketchtool import ListLayer


def draw_artboard(
        artboard_folder: str,
        artboard_json_name: str,
        artboard_export_image_name: Optional[str] = None,
        output_image_path: Optional[str] = None,
):
    artboard = ListLayer.from_json(open(path.join(artboard_folder, artboard_json_name), 'r').read())

    def dfs(layer: ListLayer, root: ListLayer, canvas: Image.Image) -> Image.Image:
        if len(layer.layers) > 0:
            for child in layer.layers:
                dfs(child, root, canvas)
        else:
            layer_image_path = path.join(artboard_folder, f"{layer.id}.png")
            if path.exists(layer_image_path):
                image = Image.open(layer_image_path).convert("RGBA")
                canvas.alpha_composite(image, (
                    int(layer.trimmed.x - root.trimmed.x), int(layer.trimmed.y - root.trimmed.y)))
        return canvas

    res = dfs(artboard, artboard,
              Image.new("RGBA", (int(artboard.rect.width), int(artboard.rect.height)), (255, 255, 255, 255)))
    if output_image_path is not None:
        res.save(output_image_path)
    if artboard_export_image_name is not None:
        real_res = Image.open(path.join(artboard_folder, artboard_export_image_name)).convert("RGBA")
        compare = Image.new("RGBA", (res.width * 2, res.height), (255, 255, 255, 255))
        compare.paste(res, (0, 0))
        compare.paste(real_res, (res.width, 0))
        compare.show()