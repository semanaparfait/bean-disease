"""Single-image prediction with the production model (thread-safe, hot-reloadable)."""
from __future__ import annotations

import os
import threading
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from . import config as C
from . import preprocessing as pp

ADVICE = {
    "angular_leaf_spot": "Angular leaf spot (Phaeoisariopsis griseola). Remove infected leaves, avoid overhead "
                         "watering, rotate crops and use certified disease-free seed.",
    "bean_rust": "Bean rust (Uromyces appendiculatus). Remove infected debris, improve air flow between plants "
                 "and consider a recommended fungicide; plant rust-resistant varieties next season.",
    "healthy": "No disease detected. Keep monitoring the crop regularly.",
}


class Predictor:
    def __init__(self, model_path=C.MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.version = None
        self._lock = threading.Lock()

    def load(self):
        from .model import load_metrics, load_model

        model = load_model(self.model_path)
        # warm-up so the first real request is not slow
        model(np.zeros((1, C.IMG_SIZE, C.IMG_SIZE, 3), dtype=np.float32), training=False)
        with self._lock:
            self.model = model
            self.version = load_metrics().get("version", "unknown")
        return self

    @property
    def ready(self) -> bool:
        return self.model is not None

    def predict(self, image) -> dict:
        """image: path, bytes or file-like. Returns class, confidence and all probabilities."""
        if not self.ready:
            self.load()
        t0 = time.perf_counter()
        x = pp.load_image(image)[None, ...]
        with self._lock:
            model, version = self.model, self.version
        probs = model(x, training=False).numpy()[0]
        idx = int(np.argmax(probs))
        cls = C.CLASS_NAMES[idx]
        return {
            "predicted_class": cls,
            "label": C.CLASS_LABELS[cls],
            "confidence": float(probs[idx]),
            "probabilities": {c: float(p) for c, p in zip(C.CLASS_NAMES, probs)},
            "advice": ADVICE[cls],
            "model_version": version,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }


_default: Predictor | None = None


def get_predictor() -> Predictor:
    global _default
    if _default is None:
        _default = Predictor().load()
    return _default


def predict_image(image) -> dict:
    return get_predictor().predict(image)


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(predict_image(sys.argv[1]), indent=2))
