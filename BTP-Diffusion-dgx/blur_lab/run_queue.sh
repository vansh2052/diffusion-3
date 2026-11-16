#!/bin/bash
# Runs prototypes one at a time from queue.txt ("<process> <steps> <seed> [samplers]"); new lines can be appended while it runs.
cd "$(dirname "$0")"
PY=/Users/armaanjotsingh/Desktop/diffusion/.venv/bin/python
touch done.txt
while true; do
  line=$(grep -v -x -F -f done.txt queue.txt | grep -v '^#' | grep -v '^$' | head -1)
  [ -z "$line" ] && { echo "queue empty $(date)"; break; }
  set -- $line
  echo "=== $(date +%H:%M:%S) start: $line"
  TF_CPP_MIN_LOG_LEVEL=2 $PY -u blur_proto.py run "$@" 2>&1 | grep -v -E "AutoGraph|Cause:|To silence|^W0000|^I0000|oneDNN|absl" > "logs/$1_s$3.log"
  grep -E "FID|Error|Traceback" "logs/$1_s$3.log"
  echo "$line" >> done.txt
done
