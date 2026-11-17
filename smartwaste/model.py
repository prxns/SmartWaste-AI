"""Serializable MobileNetV2 construction and validated independent inference."""
import json
import numpy as np
from config import CLASS_NAMES, IMAGE_SIZE, MODEL_PATH
from smartwaste.preprocessing import preprocess_image


def build_model(size=IMAGE_SIZE, weights="imagenet", alpha=1.0):
    import tensorflow as tf
    base = tf.keras.applications.MobileNetV2(
        input_shape=(size, size, 3), include_top=False, weights=weights, alpha=alpha)
    base.trainable = False
    inputs = tf.keras.Input((size, size, 3), name="preprocessed_rgb")
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax", name="waste_category")(x)
    model = tf.keras.Model(inputs, outputs, name="smartwaste_mobilenetv2")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


class WasteClassifier:
    def __init__(self, model_path=MODEL_PATH):
        model_path = __import__("pathlib").Path(model_path)
        mapping_path = model_path.parent / "class_names.json"
        if not model_path.is_file() or not mapping_path.is_file():
            raise FileNotFoundError("Trained model is unavailable. Run python -m scripts.train_model after downloading TrashNet.")
        with mapping_path.open(encoding="utf-8") as handle:
            self.class_names = json.load(handle)
        if self.class_names != list(CLASS_NAMES):
            raise ValueError("Class mapping differs from the six original TrashNet labels.")
        import tensorflow as tf
        self.model = tf.keras.models.load_model(model_path, compile=False, safe_mode=True)
        shape = self.model.input_shape
        if len(shape) != 4 or shape[1] != shape[2] or shape[-1] != 3 or self.model.output_shape[-1] != 6:
            raise ValueError("Model input/output dimensions are incompatible.")
        self.size = int(shape[1])

    def predict(self, image):
        scores = np.asarray(self.model(preprocess_image(image, self.size), training=False))[0]
        if scores.shape != (6,) or not np.all(np.isfinite(scores)) or np.any(scores < 0) or not np.isclose(scores.sum(), 1, atol=1e-3):
            raise ValueError("Model did not return six valid softmax scores.")
        order = np.argsort(-scores)
        return {"category": self.class_names[int(order[0])],
                "top_three": [{"category": self.class_names[int(i)], "score": float(scores[i])} for i in order[:3]],
                "scores": dict(zip(self.class_names, map(float, scores))),
                "input_dimensions": [self.size, self.size, 3]}
