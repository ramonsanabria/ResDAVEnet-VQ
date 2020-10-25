#!/bin/bash 

# Author: Wei-Ning Hsu

source /disk/scratch1/ramons/myenvs/p3_dave/bin/activate

python dump_hdf5_dataset.py \
  "/disk/scratch1/ramons/image_seg/data/facc/facc_dev.json" \
  "./data/facc_dev.json" \
  "./data/facc_audio_dev.hdf5" \
  "./data/facc_image_dev.hdf5" 
