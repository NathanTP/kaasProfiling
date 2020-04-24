#!/bin/bash
set -e
# Usage: ./gatherTrace.sh logfile.csv CALL_TO_TRACE
# CALL_TO_TRACE should be whatever you would normally call to run the
# application we're profiling.

LOGFILE=$1
shift
nvprof --log-file $LOGFILE --csv --print-gpu-trace "$@"
