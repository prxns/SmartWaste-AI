"""Bounded decoding and identical training/inference pixel preparation."""
from io import BytesIO
from pathlib import Path
import warnings
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from config import IMAGE_SIZE, MAX_UPLOAD_BYTES, MAX_IMAGE_PIXELS


def read_image(source, max_bytes=MAX_UPLOAD_BYTES):
    if isinstance(source, (str, Path)):
        with open(source, "rb") as handle:
            raw = handle.read(max_bytes + 1)
    elif isinstance(source, bytes):
        raw = source
    else:
        source.seek(0)
        raw = source.read(max_bytes + 1)
    if not raw or len(raw) > max_bytes:
        raise ValueError(f"Select a nonempty image no larger than {max_bytes // 1024 // 1024} MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise ValueError("Only genuine JPEG and PNG images are supported.")
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Image exceeds the 24 megapixel limit.")
                image.verify()
            with Image.open(BytesIO(raw)) as image:
                image = ImageOps.exif_transpose(image)
                if image.mode in {"RGBA", "LA", "P"}:
                    rgba = image.convert("RGBA")
                    background = Image.new("RGBA", rgba.size, "white")
                    image = Image.alpha_composite(background, rgba)
                return image.convert("RGB").copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError("This image is invalid, damaged, or too large to decode safely.") from exc


def resize_pixels(image, size=IMAGE_SIZE):
    """Return raw RGB float32 pixels in [0,255]; aspect is resized consistently."""
    return np.asarray(image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)


def preprocess_image(image, size=IMAGE_SIZE):
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    return preprocess_input(resize_pixels(image, size).copy())[None, ...]
