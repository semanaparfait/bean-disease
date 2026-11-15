#!/usr/bin/env bash
# Run the same Locust flood against 1, 2 and 3 API containers behind nginx and summarise the results.
# Usage: bash scripts/run_load_tests.sh [users] [duration]
set -e
USERS=${1:-50}; DURATION=${2:-60s}
mkdir -p locust/results
docker compose build
for N in 1 2 3; do
  echo "=== $N container(s) ==="
  docker compose up -d --scale api=$N api lb
  docker compose restart lb            # re-resolve the replicas
  until curl -sf http://localhost:8000/health | grep -q '"model_loaded":true'; do sleep 3; done
  sleep 10                             # let every replica finish loading the model
  locust -f locust/locustfile.py --host http://localhost:8000 --headless -u "$USERS" -r 10 -t "$DURATION" \
         --csv "locust/results/containers_$N" --only-summary
done
python scripts/summarize_load_tests.py
