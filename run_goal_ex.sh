#!/bin/bash
# run_commands.sh

# Execute the Python script to generate commands and capture them
mapfile -t commands < <(python sub_cluster.py)

# Execute each command
for cmd in "${commands[@]}"
do
    echo "Running command: $cmd"
    eval $cmd
done