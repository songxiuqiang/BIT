#!/bin/bash

cd config &&

if [ ! -d "moped" ]; then
    mkdir moped
    echo "Directory 'moped' created."
else
    echo "Directory 'moped' already exists."
fi

if [ ! -d "rbot" ]; then
    mkdir rbot
    echo "Directory 'rbot' created."
else
    echo "Directory 'rbot' already exists."
fi

if [ ! -d "rbot_3d3r6" ]; then
    mkdir rbot_3d3r6
    echo "Directory 'rbot_3d3r6' created."
else
    echo "Directory 'rbot_3d3r6' already exists."
fi

python config_generator.py
# python config_generator.py "moped"  # only generate moped configs
# python config_generator.py "rbot"  # only generate rbot configs
