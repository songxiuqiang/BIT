#!/bin/bash

if [ ! -d "build" ]; then
    mkdir build
    echo "Directory 'build' created."
else
    echo "Directory 'build' already exists."
fi

cd tracker &&

if [ ! -d "build" ]; then
    mkdir build
    echo "Directory 'build' created."
else
    echo "Directory 'build' already exists."
fi

if [ ! -d "bin" ]; then
    mkdir bin
    echo "Directory 'bin' created."
else
    echo "Directory 'bin' already exists."
fi

cd build &&
cmake -D CMAKE_BUILD_TYPE=Release .. && 
make -j$(nproc)