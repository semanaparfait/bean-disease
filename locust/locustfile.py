"""Flood the prediction endpoint with real test images.

Headless example:
  locust -f locust/locustfile.py --host http://localhost:8000 --headless -u 50 -r 10 -t 60s --csv locust/results/c1
Web UI:
  locust -f locust/locustfile.py --host http://localhost:8000      (open http://localhost:8089)
"""
import random
from pathlib import Path

from locust import HttpUser, between, task

TEST_DIR = Path(__file__).resolve().parent.parent / "data" / "test"
IMAGES = [(p.name, p.read_bytes()) for p in sorted(TEST_DIR.glob("*/*.jpg"))][:60]


class FarmerUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task(10)
    def predict(self):
        name, data = random.choice(IMAGES)
        with self.client.post("/predict", files={"file": (name, data, "image/jpeg")},
                              name="/predict", catch_response=True) as r:
            if r.status_code != 200 or "predicted_class" not in r.text:
                r.failure(f"bad response {r.status_code}")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
