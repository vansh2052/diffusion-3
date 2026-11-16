#!/bin/bash
# Low-power queue: efficiency cores only (taskpolicy -b), 4 TF threads. Reads queue_micro.txt; appendable while running.
cd "$(dirname "$0")"
PY=/Users/armaanjotsingh/Desktop/diffusion/.venv/bin/python
export TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2
touch done_micro.txt
while true; do
  line=$(grep -v -x -F -f done_micro.txt queue_micro.txt | grep -v '^#' | grep -v '^$' | head -1)
  [ -z "$line" ] && { echo "queue empty $(date +%H:%M:%S)"; break; }
  set -- $line
  echo "=== $(date +%H:%M:%S) start: $line"
  taskpolicy -b $PY -u blur_micro.py run "$@" 2>&1 | grep --line-buffered -v -E "AutoGraph|Cause:|To silence|^W0000|^I0000|oneDNN|absl|LinAlgWarning|covmean" > "logs_micro/$1_s$3.log"
  grep -E "KID|Error|Traceback" "logs_micro/$1_s$3.log"
  echo "$line" >> done_micro.txt
done
