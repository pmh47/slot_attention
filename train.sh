#!/usr/bin/bash

cd `dirname "$(readlink -f "$0")"`/.. # i.e. change to the directory above that containing this script
export PYTHONPATH=.
exec python slot_attention/object_discovery/train.py  --model_dir slot_attention/checkpoints/arrow --dataset ood --data_path data/arrow --resolution 96 --num_slots 5
#exec python slot_attention/object_discovery/train.py  --model_dir slot_attention/checkpoints/gqn --dataset ood --data_path data/gqn --resolution 96 --num_slots 5
#exec python slot_attention/object_discovery/train.py  --model_dir slot_attention/checkpoints/gqn_7-slot --dataset ood --data_path data/gqn --resolution 96 --num_slots 7

# For training on original clevr data (requires tensorflow_datasets installed):
#export TFDS_DATA_DIR=/workspace/tensorflow_datasets
#exec python slot_attention/object_discovery/train.py  --model_dir slot_attention/checkpoints/object_discovery
