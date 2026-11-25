"""Load test without Docker: the same Locust flood against 1, 2 and 3 uvicorn worker processes.

Each worker is an independent process with its own copy of the model (like one container each).
Usage:  python scripts/run_load_tests_local.py [users] [duration]
"""
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
USERS = sys.argv[1] if len(sys.argv) > 1 else "30"
DURATION = sys.argv[2] if len(sys.argv) > 2 else "60s"
PORT = 8001
RES = ROOT / "locust" / "results"
RES.mkdir(parents=True, exist_ok=True)


def wait_ready(timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if b'"model_loaded":true' in urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=3).read():
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError("API did not start")


for n in (1, 2, 3):
    env = {**os.environ, "DB_PATH": str(Path(tempfile.gettempdir()) / f"loadtest_{n}.db"), "AUTO_RETRAIN": "false"}
    print(f"=== {n} replica(s) ===", flush=True)
    server = subprocess.Popen([PY, "-m", "uvicorn", "api.main:app", "--port", str(PORT), "--workers", str(n)],
                              cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_ready()
        time.sleep(20 * n)  # let every worker finish loading the model
        subprocess.run([PY, "-m", "locust", "-f", "locust/locustfile.py", "--host", f"http://localhost:{PORT}", "--headless",
                        "-u", USERS, "-r", "5", "-t", DURATION, "--csv", str(RES / f"local_{n}_workers"),
                        "--only-summary"], cwd=ROOT, check=False)
    finally:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(server.pid), "/T", "/F"], capture_output=True)
        else:
            server.terminate()
        server.wait()
        time.sleep(5)

subprocess.run([PY, "scripts/summarize_load_tests.py"], cwd=ROOT)
