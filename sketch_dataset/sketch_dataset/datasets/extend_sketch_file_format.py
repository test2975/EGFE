from dataclasses import dataclass, field
from typing import Union, List, Dict, Type

from fastclasses_json import dataclass_json, JSONMixin
from sketch_document_py import sketch_file_format as sff

from sketch_dataset.sketchtool.sketchtool_wrapper import ListLayer


def to_object(obj):
    if obj is not None and '_class' in obj.keys() and (obj['_class'] in class_map.keys()):
        return class_map[obj['_class']].from_dict(obj)
    else:
        return sff.to_object(obj)


ExtendAnyLayer = Union[
    'ExtendSymbolMaster',
    'ExtendGroup',
    'ExtendOval',
    'ExtendPolygon',
    'ExtendRectangle',
    'ExtendShapePath',
    'ExtendStar',
    'ExtendTriangle',
    'ExtendShapeGroup',
    'ExtendText',
    'ExtendSymbolInstance',
    'ExtendSlice',
    'ExtendHotspot',
    'ExtendBitmap'
]


@dataclass_json
@dataclass
class _ExtendAnyGroupBase(JSONMixin):
    layers: List[ExtendAnyLayer] = field(
        metadata={'fastclasses_json': {'decoder': lambda lst: [to_object(x) for x in lst if x is not None]}},
    )


@dataclass_json
@dataclass
class ExtendSymbolMaster(_ExtendAnyGroupBase, sff.SymbolMaster, ListLayer):
    pass


@dataclass_json
@dataclass
class ExtendGroup(_ExtendAnyGroupBase, sff.Group, ListLayer):
    pass


@dataclass_json
@dataclass
class ExtendOval(sff.Oval, ListLayer):
    pass


@dataclass_json
@dataclass
class ExtendPolygon(sff.Polygon, ListLayer):
    pass


@dataclass_json
@dataclass
class ExtendRectangle(sff.Rectangle, ListLayer):
    pass


@dataclass_json
@dataclass
class ExtendShapePath(sff.ShapePath, ListLayer):
    pass


@dataclass_json
@dataclass
class ExtendStar(sff.Star, ListLayer):
    pass


@dataclass_json
@dataclass
class ExtendTriangle(sff.Triangle, ListLayer):
    pass


@dataclass_json
@dataclass
class ExtendShapeGroup(_ExtendAnyGroupBase, sff.ShapeGroup, ListLayer):
    pass


@dataclass_json
@dataclass
class ExtendText(sff.Text, ListLayer):
    pass


@dataclass_json
@dataclass
class ExtendSymbolInstance(sff.SymbolInstance, ListLayer):
    pass


@dataclass_json
@dataclass
class ExtendSlice(sff.Slice, ListLayer):
    pass


@dataclass_json
@dataclass
class ExtendHotspot(sff.Hotspot, ListLayer):
    pass


@dataclass_json
@dataclass
class ExtendBitmap(sff.Bitmap, ListLayer):
    pass


@dataclass_json
@dataclass
class ExtendArtboard(_ExtendAnyGroupBase, sff.Artboard, ListLayer):
    pass


class_map: Dict[str, Type[JSONMixin]] = {
    'shapePath': ExtendShapePath,
    'group': ExtendGroup,
    'star': ExtendStar,
    'bitmap': ExtendBitmap,
    'oval': ExtendOval,
    'symbolInstance': ExtendSymbolInstance,
    'slice': ExtendSlice,
    'polygon': ExtendPolygon,
    'triangle': ExtendTriangle,
    'shapeGroup': ExtendShapeGroup,
    'text': ExtendText,
    'rectangle': ExtendRectangle,
    'symbolMaster': ExtendSymbolMaster,
    'MSImmutableHotspotLayer': ExtendHotspot, }
