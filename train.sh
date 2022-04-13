#!/usr/bin/bash

cd `dirname "$(readlink -f "$0")"`/.. # i.e. change to the directory above that containing this script
export PYTHONPATH=.
export TFDS_DATA_DIR=/workspace/tensorflow_datasets
#exec python slot_attention/object_discovery/train.py  --model_dir slot_attention/checkpoints/object_discovery
exec python slot_attention/object_discovery/train.py  --model_dir slot_attention/checkpoints/arrow --dataset arrow --data_path data/arrow --resolution 96 --num_slots 5
