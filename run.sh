#!/bin/bash

###################
#      SETUP      #
###################

if [ ! -d "./VIBE" ] then
  echo "Installing VIBE..."
  git clone https://github.com/oli4jansen/VIBE
  ./VIBE/install.sh
fi

###################
#   EXTRACTION   #
###################

./VIBE/run.sh ./videos
mv ./VIBE/output/* ./data/motion/

###################
#    RENDERING    #
###################

if ! [ -x "$(command -v blender)" ]; then
  echo "Installing Blender.."
fi
blender -t 1 -P ./generate.py -b | grep '^\[synth_motion\]'
