from io import BytesIO
import unittest
import numpy as np
from PIL import Image
from smartwaste.preprocessing import read_image, resize_pixels


def encoded(image, format="PNG"):
    buffer=BytesIO(); image.save(buffer,format=format); return buffer.getvalue()


class PreprocessingTests(unittest.TestCase):
    def test_valid_png_and_jpeg(self):
        for format in ("JPEG","PNG"):
            image=read_image(encoded(Image.new("RGB",(20,30),"red"),format))
            self.assertEqual(image.mode,"RGB")
            self.assertEqual(image.size,(20,30))

    def test_transparency_composites_on_white(self):
        image=read_image(encoded(Image.new("RGBA",(2,2),(10,20,30,0))))
        self.assertEqual(image.getpixel((0,0)),(255,255,255))

    def test_invalid_empty_and_oversized(self):
        for raw in (b"",b"not an image",b"x"*11):
            with self.assertRaises(ValueError): read_image(raw,max_bytes=10)

    def test_unsupported_format(self):
        with self.assertRaises(ValueError): read_image(encoded(Image.new("RGB",(5,5)),"BMP"))

    def test_truncated_image(self):
        raw=encoded(Image.new("RGB",(30,30)))
        with self.assertRaises(ValueError): read_image(raw[:len(raw)//2])

    def test_raw_resize_shape_dtype_range(self):
        pixels=resize_pixels(Image.new("RGB",(15,50),(255,128,0)),96)
        self.assertEqual(pixels.shape,(96,96,3)); self.assertEqual(pixels.dtype,np.float32)
        np.testing.assert_allclose(pixels[0,0],[255,128,0])
