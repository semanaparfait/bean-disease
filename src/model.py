"""Model creation, training, evaluation and retraining.

Model: a custom convolutional neural network built and trained from scratch (random
initialisation, no ImageNet / pretrained weights). Optimisation techniques used:
  * data augmentation (flip / rotation / zoom / translation / contrast) inside the model
  * Batch Normalisation after every convolution (faster, more stable training)
  * L2 weight regularisation on every convolution and the output layer
  * Dropout (light after each conv block, stronger before the classifier)
  * Global Average Pooling instead of a large Flatten + Dense (far fewer parameters)
  * Adam optimiser, EarlyStopping (restore best weights) and ReduceLROnPlateau

Retraining loads the *custom trained model* (models/bean_disease_cnn.h5) as the
pretrained starting point and fine-tunes it on newly uploaded data mixed with a
replay sample of the original training set. The new model is only promoted to
production if it does not perform worse on the held-out test set.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)

from . import config as C
from . import database as db
from . import preprocessing as pp

tf.keras.utils.set_random_seed(C.SEED)
layers, keras = tf.keras.layers, tf.keras

retrain_lock = threading.Lock()


# ---------------------------------------------------------------- model creation
CONV_BLOCKS = [(64, 2), (128, 2), (256, 2)]   # (filters, separable convolutions) per block
# Keras default BN momentum (0.99) is far too slow with only ~32 batches per epoch: the running variance
# stays near its initial value 1.0, so in inference mode every layer shrinks the signal and the model
# predicts one class. 0.9 lets the running statistics catch up within the first epochs.
BN_MOMENTUM = 0.9


def build_model(img_size: int = C.IMG_SIZE, dropout: float = 0.4, block_dropout: float = 0.1, l2: float = 1e-4):
    """Custom CNN, randomly initialised (trained from scratch).

    A strided 3x3 "stem" convolution halves the resolution first, then three blocks of
    depthwise-separable convolutions (much cheaper than standard convolutions, so the model
    trains on a laptop CPU and serves fast on a small cloud instance).
    """
    reg = keras.regularizers.l2(l2)
    augment = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.15),
    ], name="augmentation")

    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = augment(inputs)
    x = layers.Rescaling(1.0 / 255, name="scaling")(x)          # [0, 255] -> [0, 1]
    x = layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False, kernel_regularizer=reg, name="stem_conv")(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM, name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    for b, (filters, n_convs) in enumerate(CONV_BLOCKS, start=1):
        for c in range(1, n_convs + 1):
            x = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False, pointwise_regularizer=reg,
                                       name=f"block{b}_sepconv{c}")(x)
            x = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f"block{b}_bn{c}")(x)
            x = layers.ReLU(name=f"block{b}_relu{c}")(x)
        x = layers.MaxPooling2D(name=f"block{b}_pool")(x)
        x = layers.Dropout(block_dropout, name=f"block{b}_dropout")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(len(C.CLASS_NAMES), activation="softmax", kernel_regularizer=reg,
                           name="predictions")(x)
    return keras.Model(inputs, outputs, name="bean_disease_cnn")


def freeze_batchnorm(model):
    """Keep BatchNorm statistics fixed while fine-tuning on a small batch of new data."""
    for layer in model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def compile_model(model, lr: float):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")],
    )


def callbacks(patience: int = 12, lr_patience: int = 4):
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=lr_patience, min_lr=1e-6),
    ]


# ---------------------------------------------------------------- evaluation
def evaluate(model, X: np.ndarray, y: np.ndarray) -> dict:
    probs = model.predict(pp.make_dataset(X, y), verbose=0)
    pred = probs.argmax(1)
    y_onehot = np.eye(len(C.CLASS_NAMES))[y]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "loss": float(log_loss(y_onehot, np.clip(probs, 1e-7, 1))),
        "precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y, pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_onehot, probs, average="macro", multi_class="ovr")),
        "n_samples": int(len(y)),
        "confusion_matrix": confusion_matrix(y, pred, labels=range(len(C.CLASS_NAMES))).tolist(),
        "per_class": classification_report(y, pred, target_names=C.CLASS_NAMES, output_dict=True,
                                           zero_division=0),
    }


def load_test_set():
    return pp.load_arrays(pp.list_images(C.TEST_DIR))


def save_model(model, metrics: dict, extra: dict | None = None) -> str:
    """Archive the current production model and save the new one with its metrics."""
    version = datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M%S")
    # Re-create a clean copy of the architecture and copy the weights over: models loaded from legacy .h5
    # cannot always be re-serialised to .h5 by Keras 3 ("cannot pickle 'module' object").
    clean = build_model()
    clean.set_weights(model.get_weights())
    tmp = C.MODEL_PATH.with_suffix(".tmp.h5")
    clean.save(tmp)                       # write to a temp file first so a failed save never corrupts production
    if C.MODEL_PATH.exists():
        old_version = load_metrics().get("version", "previous")
        shutil.copy2(C.MODEL_PATH, C.ARCHIVE_DIR / f"{C.MODEL_PATH.stem}_{old_version}.h5")
    os.replace(tmp, C.MODEL_PATH)
    payload = {"version": version, "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "class_names": C.CLASS_NAMES, "img_size": C.IMG_SIZE, "test_metrics": metrics, **(extra or {})}
    C.METRICS_PATH.write_text(json.dumps(payload, indent=2))
    return version


def load_metrics() -> dict:
    return json.loads(C.METRICS_PATH.read_text()) if C.METRICS_PATH.exists() else {}


def load_model(path: Path = C.MODEL_PATH):
    return keras.models.load_model(path, compile=False)


# ---------------------------------------------------------------- full offline training
def train(epochs: int = C.TRAIN_EPOCHS, verbose: int = 2):
    """Train the custom CNN from scratch on data/train (used by the CLI). Returns model, history, metrics."""
    items = pp.list_images(C.TRAIN_DIR)
    train_items, val_items = pp.stratified_split(items)
    X_tr, y_tr = pp.load_arrays(train_items)
    X_val, y_val = pp.load_arrays(val_items)
    ds_tr = pp.make_dataset(X_tr, y_tr, shuffle=True)
    ds_val = pp.make_dataset(X_val, y_val)

    model = build_model()
    compile_model(model, 1e-3)
    hist = model.fit(ds_tr, validation_data=ds_val, epochs=epochs, callbacks=callbacks(), verbose=verbose)

    X_te, y_te = load_test_set()
    metrics = evaluate(model, X_te, y_te)
    save_model(model, metrics, {"trigger": "initial_training", "n_train_images": len(items)})
    return model, hist.history, metrics


# ---------------------------------------------------------------- retraining
def retrain(trigger: str = "manual", epochs: int = C.RETRAIN_EPOCHS, run_id: int | None = None,
            on_done=None, tolerance: float = 0.02) -> dict:
    """Fine-tune the current production model on newly uploaded data (+ replay of original data).

    1. collect pending uploads from the database (already validated and stored on disk)
    2. preprocess them with the same pipeline used for training
    3. load the existing custom model as the pretrained starting point and fine-tune it
    4. evaluate on the held-out test set; promote only if accuracy >= old accuracy - tolerance
    """
    if not retrain_lock.acquire(blocking=False):
        raise RuntimeError("A retraining job is already running")
    run_id = run_id or db.start_run(trigger)
    t0 = time.time()
    try:
        pending = db.pending_uploads()
        if not pending:
            db.finish_run(run_id, "skipped", "No new uploaded data to retrain on")
            return {"run_id": run_id, "status": "skipped"}

        # 1-2. new data + replay sample of the original training data, preprocessed identically
        new_items = [(Path(u["path"]), C.CLASS_NAMES.index(u["label"])) for u in pending if Path(u["path"]).exists()]
        rng = random.Random(C.SEED)
        replay = []
        for idx, cls in enumerate(C.CLASS_NAMES):
            cls_items = [it for it in pp.list_images(C.TRAIN_DIR) if it[1] == idx]
            replay += rng.sample(cls_items, min(C.REPLAY_PER_CLASS, len(cls_items)))
        all_items = new_items + replay
        train_items, val_items = pp.stratified_split(all_items, val_fraction=0.2)
        X_tr, y_tr = pp.load_arrays(train_items)
        X_val, y_val = pp.load_arrays(val_items)

        # 3. existing custom model as the pretrained model -> fine-tune with a low learning rate
        model = load_model()
        freeze_batchnorm(model)
        compile_model(model, 1e-4)
        X_te, y_te = load_test_set()
        before = evaluate(model, X_te, y_te)
        hist = model.fit(pp.make_dataset(X_tr, y_tr, batch_size=16, shuffle=True),
                         validation_data=pp.make_dataset(X_val, y_val, batch_size=16),
                         epochs=epochs, callbacks=callbacks(patience=2, lr_patience=1),
                         class_weight=pp.class_weights(y_tr), verbose=2)

        # 4. evaluate and decide
        after = evaluate(model, X_te, y_te)
        summary = {"before": {k: before[k] for k in ("accuracy", "loss", "precision", "recall", "f1")},
                   "after": after, "epochs_run": len(hist.history["loss"]),
                   "history": {k: [float(v) for v in vals] for k, vals in hist.history.items()},
                   "duration_s": round(time.time() - t0, 1)}

        promote = after["accuracy"] >= before["accuracy"] - tolerance
        if promote:
            version = save_model(model, after, {"trigger": f"retrain:{trigger}", "retrain_run_id": run_id,
                                                "n_new_images": len(new_items)})
            msg = f"Model promoted: test accuracy {before['accuracy']:.3f} -> {after['accuracy']:.3f}"
            status = "completed"
        else:
            version = load_metrics().get("version")
            msg = (f"New model rejected (accuracy {after['accuracy']:.3f} < {before['accuracy']:.3f}); "
                   "kept current model")
            status = "rejected"

        # only now consume the uploads: move them into the training set so future full training includes them
        new_paths = {}
        for u in pending:
            src = Path(u["path"])
            dst = C.TRAIN_DIR / u["label"] / f"upload_{u['id']}_{src.name}"
            if src.exists():
                shutil.move(str(src), dst)
            new_paths[u["id"]] = str(dst)
        db.mark_uploads_used([u["id"] for u in pending], run_id, new_paths)
        db.finish_run(run_id, status, msg, summary, len(new_items), len(all_items), version)
        if on_done:
            on_done(status)
        return {"run_id": run_id, "status": status, "message": msg, "metrics": summary}
    except Exception as e:  # record failures so the UI can show them
        db.finish_run(run_id, "failed", f"{type(e).__name__}: {e}")
        raise
    finally:
        retrain_lock.release()
        import gc
        gc.collect()


if __name__ == "__main__":
    db.init_db()
    _, _, m = train()
    print(json.dumps({k: m[k] for k in ("accuracy", "loss", "precision", "recall", "f1", "roc_auc")}, indent=2))
