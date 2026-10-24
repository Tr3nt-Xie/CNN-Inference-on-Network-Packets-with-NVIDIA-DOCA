"""Read Part 3/4 image artifacts without guessing headers or frame order."""
from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np

HEIGHT, WIDTH, CHANNELS = 224, 224, 3
FRAME_BYTES = HEIGHT * WIDTH * CHANNELS


@dataclass
class Frames:
    pixels: np.ndarray
    ids: np.ndarray
    source: str


def _ids(values, count):
    array = np.asarray(values)
    if array.shape != (count,) or array.dtype.kind not in 'iu':
        raise ValueError('frame IDs must be a one-dimensional integer array, one per image')
    if np.any(array < 0) or np.any(array > 0xFFFFFFFF):
        raise ValueError('frame IDs must fit the unsigned 32-bit packet header field')
    array = array.astype(np.int64)
    if len(np.unique(array)) != count:
        raise ValueError('duplicate frame IDs are not allowed')
    return array


def load_frames(path, ids_path=None, expected_frames=16):
    """Supported: raw NHWC .bin, uint8 NHWC .npy, .npz, or numbered images.

    .npz contains `frames` and `frame_ids`. Raw/.npy order defaults to IDs
    0..N-1; an explicit JSON ID list overrides that convention. Directory
    filenames must end in an integer ID (e.g. frame_0007.png).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'image input does not exist: {path}')
    embedded_ids = None
    if path.is_dir():
        from PIL import Image
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.bin'})
        if not files:
            raise ValueError(f'no supported image files in {path}')
        images, embedded_ids = [], []
        for file in files:
            match = re.search(r'(\d+)$', file.stem)
            if match is None:
                raise ValueError(f'{file.name}: filename must end in the frame ID')
            embedded_ids.append(int(match.group(1)))
            if file.suffix.lower() == '.bin':
                raw = file.read_bytes()
                if len(raw) != FRAME_BYTES:
                    raise ValueError(f'{file}: expected exactly {FRAME_BYTES} bytes for one RGB image')
                image = np.frombuffer(raw, dtype=np.uint8).reshape(HEIGHT, WIDTH, CHANNELS)
            else:
                with Image.open(file) as opened:
                    image = np.asarray(opened.convert('RGB'))
            if image.shape != (HEIGHT, WIDTH, CHANNELS):
                raise ValueError(f'{file}: expected 224 x 224 RGB, got {image.shape}; do not resize during loading')
            images.append(image)
        pixels = np.stack(images)
    elif path.suffix.lower() == '.bin':
        raw = path.read_bytes()
        if not raw or len(raw) % FRAME_BYTES:
            raise ValueError(f'{path}: raw .bin must contain a nonzero multiple of {FRAME_BYTES} bytes, with no headers')
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(-1, HEIGHT, WIDTH, CHANNELS).copy()
    elif path.suffix.lower() == '.npy':
        pixels = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as archive:
            if not {'frames', 'frame_ids'} <= set(archive.files):
                raise ValueError(f'{path}: .npz requires arrays named frames and frame_ids')
            pixels, embedded_ids = archive['frames'], archive['frame_ids']
    else:
        raise ValueError(f'unsupported image input: {path}; use .bin, .npy, .npz, or a numbered image directory')

    if pixels.dtype != np.uint8 or pixels.ndim != 4 or pixels.shape[1:] != (HEIGHT, WIDTH, CHANNELS):
        raise ValueError(f'{path}: expected uint8 (N, 224, 224, 3), got {pixels.dtype} {pixels.shape}')
    count = len(pixels)
    if count == 0 or (expected_frames is not None and count != expected_frames):
        raise ValueError(f'{path}: expected {expected_frames or "at least one"} frames, got {count}')
    if ids_path is not None:
        if embedded_ids is not None:
            raise ValueError('ID sidecars apply only to raw .bin/.npy; directory/.npz inputs already identify frames')
        embedded_ids = json.loads(Path(ids_path).read_text(encoding='utf-8'))
    ids = _ids(np.arange(count) if embedded_ids is None else embedded_ids, count)
    order = np.argsort(ids)
    return Frames(np.ascontiguousarray(pixels[order]), ids[order], str(path.resolve()))


def compare_pixels(original, received):
    if not np.array_equal(original.ids, received.ids):
        missing = sorted(set(original.ids.tolist()) - set(received.ids.tolist()))
        extra = sorted(set(received.ids.tolist()) - set(original.ids.tolist()))
        raise ValueError(f'frame ID sets differ: missing={missing}, extra={extra}')
    return np.count_nonzero(original.pixels != received.pixels, axis=(1, 2, 3))
