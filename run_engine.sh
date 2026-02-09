#!/bin/bash

# 1. Ensure the working directory is set to the script's location (the engines directory)
# This guarantees relative paths for models and assets remain valid.
cd "$(dirname "$0")"

# 2. Log execution metadata and runtime errors for debugging purposes
# Essential for diagnosing unexpected exits or segmentation faults.
echo "Engine process initialized at $(date)" >> .engine_boot.log

# 3. Execute the Engine binary (using 'exec' to replace the current shell process)
# Optimizes resource allocation by handing over the PID directly to the engine.
exec ./kiy_engine_v5_alpha
