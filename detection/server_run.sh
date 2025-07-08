#!/bin/bash

NUM_ROUND=5
MIN_FIT_CLIENTS=3
MIN_AVAL_CLIENTS=3

echo "===================================="
echo "[INFO] Starting detection server"
echo "       Rounds           : $NUM_ROUND"
echo "       Min Fit Clients  : $MIN_FIT_CLIENTS"
echo "       Min Aval Clients : $MIN_AVAL_CLIENTS"
echo "===================================="

python detection_server.py $NUM_ROUND $MIN_FIT_CLIENTS $MIN_AVAL_CLIENTS
