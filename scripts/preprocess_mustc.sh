#!/bin/bash 

# Author: Wei-Ning Hsu

source /disk/scratch1/ramons/myenvs/p3_dave/bin/activate

python dump_hdf5_dataset.py \
  "/disk/scratch1/ramons/image_seg/data/mustc/mustc_dev_en-fr.json" \
  "/disk/scratch1/ramons/image_seg/ResDAVEnet-VQ/data/mustc/enfr/mustc_dev.json" \
  "/disk/scratch1/ramons/image_seg/ResDAVEnet-VQ/data/mustc/enfr/mustc_audio_dev.hdf5" \
  "/disk/scratch1/ramons/image_seg/ResDAVEnet-VQ/data/mustc/enfr/mustc_image_dev.hdf5" 
