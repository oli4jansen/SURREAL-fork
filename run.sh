#!/bin/bash

###################
#      SETUP      #
###################

git clone https://github.com/oli4jansen/VIBE
./VIBE/install.sh

###################
#   EXTRACTION   #
###################

./VIBE/run.sh ./videos
mv ./VIBE/output/* ./data/motion/

###################
#    RENDERING    #
###################

blender -t 1 -P ./generate.py -b | grep '^\[synth_motion\]'
