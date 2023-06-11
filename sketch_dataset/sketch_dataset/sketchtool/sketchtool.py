from enum import Enum
from os import path
from asyncio.subprocess import create_subprocess_shell, Process, PIPE
from typing import List, Union, Optional, Tuple, Any, Callable


class BackgroundColor:

    def __new__(cls, r: int, g: int, b: int):
        obj = object.__new__(cls)
        obj.r = min(255, max(0, r))
        obj.g = min(255, max(0, g))
        obj.b = min(255, max(0, b))
        return obj

    def __str__(self):
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"


class BackgroundColorEnum(BackgroundColor, Enum):
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    PURPLE = (255, 0, 255)
    CYAN = (0, 255, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)


class ExportFormat(Enum):
    PNG = "png"
    JPG = "jpg"
    PDF = "pdf"
    EPS = "eps"
    SVG = "svg"


def bool_to_cmd(value: bool):
    return "YES" if value else "NO"


class SketchTool:
    SKETCHTOOL_RELATIVE_PATH = "/Applications/Sketch.app/Contents/MacOS/sketchtool"

    def __init__(self, sketch_path: str):
        self.sketch_path = sketch_path
        self.export = self.Export(self)
        self.list = self.List(self)

    @property
    def executable(self):
        return path.join(self.sketch_path, SketchTool.SKETCHTOOL_RELATIVE_PATH)

    async def execute(self, command: str) -> Process:
        return await create_subprocess_shell(f"{self.executable} {command}", stdout=PIPE, stderr=PIPE)

    class Export:
        def __init__(self, sketchtool: "SketchTool"):
            self.sketchtool = sketchtool

        async def layers(
                self,
                document: str,
                output: Optional[str] = None,
                formats: Optional[List[ExportFormat]] = None,
                items: Optional[List[str]] = None,
                item: Optional[str] = None,
                scales: Optional[List[Union[int, float]]] = None,
                save_for_web: Optional[bool] = None,
                overwriting: Optional[bool] = None,
                trimmed: Optional[bool] = None,
                background: Optional[BackgroundColor] = None,
                group_contents_only: Optional[bool] = None,
                use_id_for_name: Optional[bool] = None,
                suffixing: Optional[bool] = None,
        ) -> Process:
            cmd = f"export layers \"{document}\""
            args = [
                ("output", output, lambda x: f"\"{x}\""),
                ("formats", formats, lambda x: ",".join([f.value for f in x])),
                ("items", items, lambda x: ",".join(x)),
                ("item", item, lambda x: x),
                ("scales", scales, lambda x: ",".join([str(s) for s in x])),
                ("save-for-web", save_for_web, bool_to_cmd),
                ("overwriting", overwriting, bool_to_cmd),
                ("trimmed", trimmed, bool_to_cmd),
                ("background", background, str),
                ("group-contents-only", group_contents_only, bool_to_cmd),
                ("use-id-for-name", use_id_for_name, bool_to_cmd),
                ("suffixing", suffixing, bool_to_cmd),
            ]
            for arg in args:
                if arg[1] is not None:
                    cmd += f" --{arg[0]}={arg[2](arg[1])}"
            return await self.sketchtool.execute(
                cmd
            )

        async def artboards(
                self,
                document: str,
                output: Optional[str] = None,
                formats: Optional[List[ExportFormat]] = None,
                items: Optional[List[str]] = None,
                item: Optional[str] = None,
                scales: Optional[List[Union[int, float]]] = None,
                save_for_web: Optional[bool] = None,
                overwriting: Optional[bool] = None,
                trimmed: Optional[bool] = None,
                group_contents_only: Optional[bool] = None,
                use_id_for_name: Optional[bool] = None,
                include_symbols: Optional[bool] = None,
                export_page_as_fallback: Optional[bool] = None,
                serial: Optional[bool] = None,
        ) -> Process:
            cmd = f"export artboards \"{document}\""
            args: List[Tuple[str, Any, Callable[[Any], str]]] = [
                ("output", output, lambda x: f"\"{x}\""),
                ("formats", formats, lambda x: ",".join([f.value for f in x])),
                ("items", items, lambda x: ",".join(x)),
                ("item", item, lambda x: x),
                ("scales", scales, lambda x: ",".join([str(s) for s in x])),
                ("save-for-web", save_for_web, bool_to_cmd),
                ("overwriting", overwriting, bool_to_cmd),
                ("trimmed", trimmed, bool_to_cmd),
                ("group-contents-only", group_contents_only, bool_to_cmd),
                ("use-id-for-name", use_id_for_name, bool_to_cmd),
                ("include-symbols", include_symbols, bool_to_cmd),
                ("export-page-as-fallback", export_page_as_fallback, bool_to_cmd),
                ("serial", serial, bool_to_cmd),
            ]
            for arg in args:
                if arg[1] is not None:
                    cmd += f" --{arg[0]}={arg[2](arg[1])}"
            return await self.sketchtool.execute(cmd)

    class List:
        def __init__(self, sketchtool: "SketchTool"):
            self.sketchtool = sketchtool

        async def layers(self, document: str) -> Process:
            cmd = f"list layers \"{document}\""
            return await self.sketchtool.execute(cmd)

        async def artboards(self, document: str) -> Process:
            cmd = f"list artboards \"{document}\""
            return await self.sketchtool.execute(cmd)


__all__ = ["BackgroundColor", "BackgroundColorEnum", "ExportFormat", "SketchTool"]
