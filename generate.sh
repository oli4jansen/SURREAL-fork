#!/bin/bash

# Run Python script through Blender
blender -t 1 -P generate.py | grep '^\[synth_motion\]' -b

# -b