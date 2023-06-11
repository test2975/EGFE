# encoding: utf-8
import os
import struct
from typing import Tuple


def get_png_size(file_path) -> Tuple[int, int]:
    """
    Return (width, height) for a given img file content - no external
    dependencies except the os and struct modules from core
    """
    size = os.path.getsize(file_path)

    with open(file_path, 'rb') as input:
        height = -1
        width = -1
        data = input.read(25)

        if ((size >= 24) and data.startswith(b'\211PNG\r\n\032\n')
                and (data[12:16] == b'IHDR')):
            # PNGs
            w, h = struct.unpack(">LL", data[16:24])
            width = int(w)
            height = int(h)

    return width, height
