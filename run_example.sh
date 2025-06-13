#!/bin/bash

# if stdc++ can not be found, try to export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 && CUDA_VISIBLE_DEVICES=0 python examples/run_example.py $1 $2

# else
CUDA_VISIBLE_DEVICES=0 python examples/run_example.py $1 $2