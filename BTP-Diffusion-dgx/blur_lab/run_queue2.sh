#!/bin/bash
# CPU-limited queue: 4 TF threads, low priority. Reads queue2.txt ("<process> <steps> <seed> [samplers]"), appendable while running.
cd "$(dirname "$0")"
PY=/Users/armaanjotsingh/Desktop/diffusion/.venv/bin/python
export TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=2 TF_CPP_MIN_LOG_LEVEL=2
touch done.txt
while true; do
  line=$(grep -v -x -F -f done.txt queue2.txt | grep -v '^#' | grep -v '^$' | head -1)
  [ -z "$line" ] && { echo "queue empty $(date +%H:%M:%S)"; break; }
  set -- $line
  echo "=== $(date +%H:%M:%S) start: $line"
  nice -n 15 $PY -u blur_proto.py run "$@" 2>&1 | grep --line-buffered -v -E "AutoGraph|Cause:|To silence|^W0000|^I0000|oneDNN|absl" > "logs/$1_s$3.log"
  grep -E "FID|Error|Traceback" "logs/$1_s$3.log"
  echo "$line" >> done.txt
done
