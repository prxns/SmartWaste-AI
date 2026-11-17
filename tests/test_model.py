import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from config import CLASS_NAMES,MODEL_PATH
from smartwaste.model import WasteClassifier

HAS_TF=importlib.util.find_spec("tensorflow") is not None


class ModelValidationTests(unittest.TestCase):
    def test_missing_model_clear_error_without_tensorflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError): WasteClassifier(Path(tmp)/"missing.keras")

    def test_invalid_class_mapping_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/"bad.keras").write_bytes(b"not a model")
            (root/"class_names.json").write_text(json.dumps(list(reversed(CLASS_NAMES))))
            with self.assertRaises(ValueError): WasteClassifier(root/"bad.keras")

    @unittest.skipUnless(HAS_TF,"TensorFlow is unavailable in this environment")
    def test_preprocessing_uses_correct_mobilenet_range(self):
        from smartwaste.preprocessing import preprocess_image
        pixels=preprocess_image(Image.new("RGB",(8,8),(0,127,255)),96)
        self.assertEqual(pixels.shape,(1,96,96,3))
        np.testing.assert_allclose(pixels[0,0,0],[-1,127/127.5-1,1],atol=1e-6)

    @unittest.skipUnless(HAS_TF,"TensorFlow is unavailable in this environment")
    def test_real_architecture_serialization_and_inference(self):
        from smartwaste.model import build_model
        model=build_model(size=96,weights=None)
        self.assertFalse(model.layers[1].trainable)
        self.assertEqual(model.output_shape,(None,6))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); model.save(root/"test.keras")
            (root/"class_names.json").write_text(json.dumps(list(CLASS_NAMES)))
            classifier=WasteClassifier(root/"test.keras")
            result=classifier.predict(Image.new("RGB",(20,30)))
            self.assertEqual(len(result["top_three"]),3)
            self.assertEqual(result["input_dimensions"],[96,96,3])
            self.assertAlmostEqual(sum(result["scores"].values()),1,places=5)
            self.assertEqual(sorted(result["scores"]),list(CLASS_NAMES))

    @unittest.skipUnless(HAS_TF and MODEL_PATH.is_file(),"Trained artifact unavailable; run training first")
    def test_shipped_model_loads_and_matches_recorded_provenance(self):
        from config import RESULTS_DIR
        from smartwaste.evaluation import file_sha256
        classifier=WasteClassifier()
        self.assertIn(classifier.predict(Image.new("RGB",(50,50)))["category"],CLASS_NAMES)
        metrics=json.loads((RESULTS_DIR/"metrics.json").read_text())
        self.assertEqual(metrics["model_sha256"],file_sha256(MODEL_PATH))
        self.assertGreater(metrics["test_samples"],0)
