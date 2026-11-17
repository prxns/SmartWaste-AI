"""Re-evaluate only the persisted held-out test split, verifying image hashes."""
import json
import pandas as pd
from config import DATA_DIR, MODEL_DIR, RESULTS_DIR
from smartwaste.dataset import inventory
from smartwaste.evaluation import write_evaluation
from scripts.train_model import make_dataset


def main():
    import tensorflow as tf
    splits = pd.read_csv(RESULTS_DIR/"splits.csv")
    current = inventory(DATA_DIR).set_index("path")
    if not all(p in current.index and current.loc[p, "sha256"] == h
               for p,h in zip(splits.path, splits.sha256)):
        raise ValueError("Dataset no longer matches the persisted split manifest.")
    test = splits[splits.split == "test"]
    config = json.loads((MODEL_DIR/"training_config.json").read_text())
    model = tf.keras.models.load_model(MODEL_DIR/"smartwaste.keras", compile=False)
    scores = model.predict(make_dataset(test, DATA_DIR, config["image_size"], config["batch_size"]), verbose=0)
    metrics = write_evaluation(test.label.to_numpy(), scores, RESULTS_DIR,
                               json.loads((RESULTS_DIR/"history.json").read_text()),
                               MODEL_DIR/"smartwaste.keras", RESULTS_DIR/"splits.csv")
    print(json.dumps({k:v for k,v in metrics.items() if isinstance(v,(int,float))}, indent=2))


if __name__ == "__main__":
    main()
