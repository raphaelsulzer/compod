#!/bin/bash
set -e

# Installation script for Anaconda3 environments
echo "____________ Clean build folders and pycache _____________"
echo

rm -rf build
rm -rf pycompod.egg-info
rm -rf __pycache__
rm -rf compose/build
