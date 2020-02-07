#!/bin/bash

# Run Python script through Blender
/Applications/blender/blender.app/Contents/MacOS/blender -t 1 -P generate.py | grep '^\[synth_motion\]' -b

# -b