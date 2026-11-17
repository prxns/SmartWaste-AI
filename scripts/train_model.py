"""Train a genuine ImageNet MobileNetV2 head; never use test data to select it."""
import argparse
import json
import os
from pathlib import Path
import platform
from config import CLASS_NAMES, DATA_DIR, MODEL_DIR, RESULTS_DIR, SEED, IMAGE_SIZE
from smartwaste.dataset import inventory, split_dataset


def make_dataset(frame, root, size, batch_size, training=False, seed=SEED):
    import tensorflow as tf
    import numpy as np
    from smartwaste.preprocessing import read_image, resize_pixels
    # A small TrashNet array is bounded (~0.8 GB at 160px), avoiding Python
    # generators and keeping decoding identical to inference.
    pixels = np.stack([resize_pixels(read_image(root/p), size) for p in frame.path]).astype("uint8")
    labels = frame.label.to_numpy(dtype="int32")
    ds = tf.data.Dataset.from_tensor_slices((pixels, labels))
    if training:
        ds = ds.shuffle(len(frame), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=seed),
        tf.keras.layers.RandomRotation(0.08, fill_mode="reflect", seed=seed+1),
        tf.keras.layers.RandomZoom(0.10, fill_mode="reflect", seed=seed+2),
    ]) if training else None

    def transform(x, y):
        x = tf.cast(x, tf.float32)
        if augmentation is not None:
            x = augmentation(x, training=True)
        return tf.keras.applications.mobilenet_v2.preprocess_input(x), y

    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    return ds.map(transform, num_parallel_calls=1).with_options(options).prefetch(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size not in (96, 128, 160, 192, 224) or args.threads < 1:
        parser.error("Use positive epochs, batch size, threads and a supported MobileNetV2 resolution.")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    import tensorflow as tf
    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight
    from smartwaste.model import build_model
    from smartwaste.evaluation import write_evaluation, file_sha256
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    frame = inventory(args.data)
    splits = split_dataset(frame, args.seed)
    train = splits[splits.split == "train"]
    validation = splits[splits.split == "validation"]
    test = splits[splits.split == "test"]
    MODEL_DIR.mkdir(exist_ok=True); RESULTS_DIR.mkdir(exist_ok=True)
    splits.to_csv(RESULTS_DIR/"splits.csv", index=False)
    frame.category.value_counts().sort_index().rename("count").to_csv(RESULTS_DIR/"class_distribution.csv")
    weights = compute_class_weight("balanced", classes=np.arange(6), y=train.label)
    training_config = {**vars(args), "data": "data/dataset-resized", "architecture": "MobileNetV2",
                       "weights": "imagenet", "alpha": 1.0, "base_frozen": True,
                       "dropout": 0.3, "optimizer": "Adam", "learning_rate": 0.001,
                       "preprocessing": "tf.keras.applications.mobilenet_v2.preprocess_input",
                       "class_weights": {str(i): float(w) for i,w in enumerate(weights)},
                       "split_counts": splits.split.value_counts().to_dict(),
                       "python": platform.python_version(), "tensorflow": tf.__version__}
    (MODEL_DIR/"training_config.json").write_text(json.dumps(training_config, indent=2), encoding="utf-8")
    (MODEL_DIR/"class_names.json").write_text(json.dumps(list(CLASS_NAMES), indent=2), encoding="utf-8")
    model = build_model(args.image_size)
    train_ds = make_dataset(train, args.data, args.image_size, args.batch_size, True, args.seed)
    val_ds = make_dataset(validation, args.data, args.image_size, args.batch_size)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
                 tf.keras.callbacks.ModelCheckpoint(str(MODEL_DIR/"smartwaste.keras"), monitor="val_loss", save_best_only=True),
                 tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
                 tf.keras.callbacks.CSVLogger(str(RESULTS_DIR/"training_log.csv"))]
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                        class_weight=dict(enumerate(weights)), callbacks=callbacks, verbose=2).history
    # Checkpoint represents the best validation epoch, not the last epoch.
    model = tf.keras.models.load_model(MODEL_DIR/"smartwaste.keras", compile=False)
    test_ds = make_dataset(test, args.data, args.image_size, args.batch_size)
    scores = model.predict(test_ds, verbose=0)
    metrics = write_evaluation(test.label.to_numpy(), scores, RESULTS_DIR, history,
                               MODEL_DIR/"smartwaste.keras", RESULTS_DIR/"splits.csv")
    manifest = {"model_sha256": file_sha256(MODEL_DIR/"smartwaste.keras"),
                "class_names": list(CLASS_NAMES), "test_samples": len(test),
                "best_epoch": int(np.argmin(history["val_loss"])+1),
                "actual_epochs": len(history["loss"])}
    (MODEL_DIR/"model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in metrics.items() if isinstance(v,(int,float))}, indent=2))


if __name__ == "__main__":
    main()
