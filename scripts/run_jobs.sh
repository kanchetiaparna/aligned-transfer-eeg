#!/bin/bash
# usage: run_jobs.sh jobs.txt [workers_gpu0] [workers_gpu1]
# Each line of jobs.txt = args for train.py. Finished jobs are skipped by train.py itself.
# Jobs are pulled from one shared queue; a failed job (e.g. CUDA OOM because the GPUs are shared)
# is retried up to 4 times with a pause in between.
JOBS=$1; W0=${2:-2}; W1=${3:-10}
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
SP=${TMPDIR:-/tmp}
LOCK=$SP/neuroai_queue.lock
QUEUE=$(mktemp -p $SP queue.XXXX)
cp "$JOBS" "$QUEUE"

next_job() {  # atomically pop the first line of the queue
  flock "$LOCK" bash -c "head -n1 '$QUEUE'; sed -i '1d' '$QUEUE'"
}

worker() {
  local g=$1
  while true; do
    job=$(next_job); [ -z "$job" ] && break
    for try in 1 2 3 4 5; do
      CUDA_VISIBLE_DEVICES=$g ${PYTHON:-python} ${SCRIPT:-train.py} $job >> ../results/log_gpu$g.txt 2>&1 && break
      echo "RETRY $try gpu$g: $job" >> ../results/log_gpu$g.txt; sleep 120
    done
  done
}

for i in $(seq 1 $W0); do worker 0 & done
for i in $(seq 1 $W1); do worker 1 & done
wait; rm -f "$QUEUE"; echo "ALL DONE $JOBS"
