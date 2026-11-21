#!/bin/bash
# usage: add_workers.sh workers_gpu0 workers_gpu1
# Adds extra workers to the queues of an already running run_jobs.sh (same lock, same queue files).
# Joins the current queue; when it is empty, waits for the next queue file the runner creates
# (e.g. finetune after pretrain) and joins that one too. Stops when no runner is left.
W0=${1:-2}; W1=${2:-4}
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
SP=${TMPDIR:-/tmp}
LOCK=$SP/neuroai_queue.lock

pop() { flock "$LOCK" bash -c "[ -f '$1' ] && head -n1 '$1' && sed -i '1d' '$1'"; }

worker() {
  local g=$1 seen=""
  while pgrep -u "$USER" -f "[r]un_jobs.sh" >/dev/null; do
    Q=$(ls -t $SP/queue.???? 2>/dev/null | grep -v 'queue.lock$' | head -1)
    if [ -z "$Q" ] || [ "$Q" = "$seen" ]; then sleep 20; continue; fi
    while job=$(pop "$Q"); [ -n "$job" ]; do
      for try in 1 2 3 4 5; do
        CUDA_VISIBLE_DEVICES=$g ${PYTHON:-python} ${SCRIPT:-train.py} $job >> ../results/log_gpu$g.txt 2>&1 && break
        echo "RETRY $try gpu$g: $job" >> ../results/log_gpu$g.txt; sleep 120
      done
    done
    seen=$Q
  done
}
for i in $(seq 1 $W0); do worker 0 & done
for i in $(seq 1 $W1); do worker 1 & done
wait; echo "EXTRA WORKERS DONE"
